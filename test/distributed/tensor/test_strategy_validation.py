# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for the strategy validation library.

These tests verify that the validation logic correctly detects:
- Incorrect rules (DTensor claims valid but produces wrong results)
- Missing rules (ground truth valid but DTensor has no rule)
"""

import torch
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor._ops.strategy_validation import (
    _create_partial_input,
    extract_tensors_from_sample,
    get_1d_input_placements_for_tensor,
    get_1d_output_placements_for_tensor,
    is_fully_replicated,
    parse_placement,
    placement_tuple_to_str,
    PlacementCombination,
    validate_combination,
)
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


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
        self.assertEqual(s, "(S0, R)")

        s = placement_tuple_to_str((Partial("sum"),))
        self.assertEqual(s, "(P(sum))")

    def test_is_fully_replicated(self):
        self.assertTrue(is_fully_replicated((Replicate(),)))
        self.assertTrue(is_fully_replicated((Replicate(), Replicate())))
        self.assertFalse(is_fully_replicated((Shard(0),)))
        self.assertFalse(is_fully_replicated((Replicate(), Shard(0))))
        self.assertFalse(is_fully_replicated((Partial("sum"),)))


class TestPlacementEquivalence(TestCase):
    """Test placement equivalence utilities."""

    def test_same_placements_equivalent(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            placements_equivalent,
        )

        # Same types are equivalent
        self.assertTrue(placements_equivalent(Replicate(), Replicate(), (4, 3)))
        self.assertTrue(placements_equivalent(Shard(0), Shard(0), (4, 3)))
        self.assertTrue(placements_equivalent(Partial("sum"), Partial("sum"), (4, 3)))

    def test_different_shards_not_equivalent(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            placements_equivalent,
        )

        # Different shard dims are not equivalent (for normal sizes)
        self.assertFalse(placements_equivalent(Shard(0), Shard(1), (4, 3)))

    def test_shard_replicate_equivalent_for_size_1(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            placements_equivalent,
        )

        # Shard on size-1 dim is equivalent to Replicate
        self.assertTrue(placements_equivalent(Shard(0), Replicate(), (1, 3)))
        self.assertTrue(placements_equivalent(Replicate(), Shard(0), (1, 3)))
        self.assertTrue(placements_equivalent(Shard(1), Replicate(), (4, 1)))

        # But not when dim is not size 1
        self.assertFalse(placements_equivalent(Shard(0), Replicate(), (4, 3)))

    def test_two_size1_shards_equivalent(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            placements_equivalent,
        )

        # Two shards on different size-1 dims are equivalent
        self.assertTrue(placements_equivalent(Shard(0), Shard(1), (1, 1)))

    def test_has_equivalent_rule(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            has_equivalent_rule,
        )

        # For transpose output [2, 1], S(1) and R are equivalent
        rules = {(("S(0)",), "S(1)")}
        combo = (("S(0)",), "R")
        input_shapes = ((1, 2),)
        output_shape = (2, 1)

        self.assertTrue(has_equivalent_rule(combo, rules, input_shapes, output_shape))

        # But not for output [2, 3] where S(1) != R
        output_shape_normal = (2, 3)
        self.assertFalse(
            has_equivalent_rule(combo, rules, input_shapes, output_shape_normal)
        )


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
        # Should have Replicate + 2 Shard + 2 Partial (min/max only)
        self.assertEqual(len(placements), 5)

    def test_get_1d_output_placements_scalar(self):
        t = torch.tensor(1.0)
        placements = get_1d_output_placements_for_tensor(t)
        # Should have only Replicate (no Shard or Partial for scalars)
        self.assertEqual(len(placements), 1)
        self.assertIsInstance(placements[0], Replicate)


class TestExtractTensors(TestCase):
    """Test tensor extraction from samples."""

    def test_extract_single_input(self):
        sample = SampleInput(torch.randn(4, 3))
        tensors = extract_tensors_from_sample(sample)
        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0][0], "input")

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


class TestValidateCombination(DTensorTestBase):
    """Test the core validate_combination function."""

    @property
    def world_size(self):
        return 2

    @with_comms
    def test_valid_shard_combination(self):
        """Test that valid sharding is detected as valid."""
        # For add: S(0), S(0) -> S(0) should be valid
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Shard(0), Shard(0)), output_placement=Shard(0)
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )
        self.assertTrue(is_valid, msg)

    @with_comms
    def test_invalid_shard_combination(self):
        """Test that invalid sharding is detected as invalid."""
        # For add: S(0), S(1) -> S(0) should be invalid
        # (broadcasting doesn't work this way)
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Shard(0), Shard(1)), output_placement=Shard(0)
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )
        self.assertFalse(is_valid)

    @with_comms
    def test_valid_partial_sum(self):
        """Test that P(sum) + P(sum) -> P(sum) is valid for add."""
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Partial("sum"), Partial("sum")),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )
        self.assertTrue(is_valid, msg)

    @with_comms
    def test_invalid_partial_combination(self):
        """Test that P(sum) + P(max) -> P(sum) is invalid for add."""
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Partial("sum"), Partial("max")),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )
        self.assertFalse(is_valid)


class TestCreatePartialInput(TestCase):
    """Test the _create_partial_input helper."""

    def test_partial_sum_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        local_tensor = _create_partial_input(tensor, Partial("sum"), world_size=2)

        # Sum of local tensors should equal original
        total = sum(local_tensor._local_tensors[r] for r in range(2))
        self.assertTrue(torch.allclose(total, tensor))

    def test_partial_avg_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        local_tensor = _create_partial_input(tensor, Partial("avg"), world_size=2)

        # Average of local tensors should equal original
        total = sum(local_tensor._local_tensors[r] for r in range(2))
        avg = total / 2
        self.assertTrue(torch.allclose(avg, tensor))

    def test_partial_min_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        local_tensor = _create_partial_input(tensor, Partial("min"), world_size=2)

        # Min of local tensors should equal original
        stacked = torch.stack([local_tensor._local_tensors[r] for r in range(2)])
        result = stacked.min(dim=0).values
        self.assertTrue(torch.allclose(result, tensor))

    def test_partial_max_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        local_tensor = _create_partial_input(tensor, Partial("max"), world_size=2)

        # Max of local tensors should equal original
        stacked = torch.stack([local_tensor._local_tensors[r] for r in range(2)])
        result = stacked.max(dim=0).values
        self.assertTrue(torch.allclose(result, tensor))


if __name__ == "__main__":
    run_tests()
