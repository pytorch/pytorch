# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for the strategy validation library.

These tests verify that the validation logic correctly detects:
- Incorrect rules (DTensor claims valid but produces wrong results)
- Missing rules (ground truth valid but DTensor has no rule)
"""

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor._ops.strategy_validation import (
    _create_partial_input,
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
    PlacementCombination,
    query_single_dim_strategy,
    validate_combination,
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


class TestValidateCombination(TestCase):
    """Test the core validate_combination function."""

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

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

    def test_exhaustive_binary_op_rules(self):
        """
        Exhaustively test all placement combinations for binary ops.

        For each op, we define the complete set of valid rules. The test then:
        1. Verifies all listed rules are detected as valid
        2. Verifies all unlisted combinations are detected as invalid
        """

        # Concise placement notation: S0=Shard(0), S1=Shard(1), R=Replicate,
        # Psum=Partial(sum), Pmax=Partial(max)
        def p(s):
            """Parse concise placement string."""
            if s == "R":
                return Replicate()
            if s.startswith("S"):
                return Shard(int(s[1:]))
            if s == "Psum":
                return Partial("sum")
            if s == "Pmax":
                return Partial("max")
            raise ValueError(f"Unknown placement: {s}")

        def parse_rule(rule_str):
            """Parse 'S0,S0->S0' into ((Shard(0), Shard(0)), Shard(0))."""
            inputs_str, output_str = rule_str.split("->")
            inputs = tuple(p(s.strip()) for s in inputs_str.split(","))
            return (inputs, p(output_str.strip()))

        # Valid rules for 2D binary ops with shape (8, 4)
        # Format: "input1,input2->output"
        # NOTE: These rules are for ground-truth validation (no redistribution).
        # Only placements that produce compatible local tensor shapes are valid.
        # - Shard(d) produces chunks (4,4) for d=0 or (8,2) for d=1
        # - Replicate and Partial produce full tensors (8,4)
        # So Shard can only pair with same-dim Shard, not with R or P.
        #
        # For Partial:
        # - P(sum) + P(sum) -> P(sum) works for add (sums distribute)
        # - R * P(sum) -> P(sum) works for mul (multiplication distributes into sum)
        # - P(sum) / R -> P(sum) works for div
        # - P(max) + P(max) -> P(max) works for max (max is idempotent)
        # But NOT:
        # - R + P(sum) -> P(sum) for add (R gets added on each rank, then summed)
        VALID_RULES = {
            torch.add: [
                # Same-dim sharding (chunks match)
                "S0,S0->S0",
                "S1,S1->S1",
                # Partial sum + Partial sum -> Partial sum
                # (a0+a1) + (b0+b1) = (a0+b0) + (a1+b1) where ai/bi are per-rank
                "Psum,Psum->Psum",
                # Partial max with Replicate (NOT Pmax+Pmax, offsets accumulate)
                "Pmax,R->Pmax",
                "R,Pmax->Pmax",
            ],
            torch.mul: [
                # Same-dim sharding
                "S0,S0->S0",
                "S1,S1->S1",
                # Partial sum * Replicate -> Partial sum (multiplicative linearity)
                # r * (p0+p1) = r*p0 + r*p1 where pi are per-rank
                "Psum,R->Psum",
                "R,Psum->Psum",
            ],
            torch.div: [
                # Same-dim sharding
                "S0,S0->S0",
                "S1,S1->S1",
                # Partial sum / Replicate -> Partial sum
                # (p0+p1) / r = p0/r + p1/r
                "Psum,R->Psum",
            ],
            torch.maximum: [
                # Same-dim sharding
                "S0,S0->S0",
                "S1,S1->S1",
                # Partial max + Partial max -> Partial max
                # max(max(a0,a1), max(b0,b1)) = max(max(a0,b0), max(a1,b1))
                "Pmax,Pmax->Pmax",
                # Partial max with Replicate:
                # max(r, max(p0,p1)) = max(max(r,p0), max(r,p1)) ✓
                "Pmax,R->Pmax",
                "R,Pmax->Pmax",
            ],
        }

        # All possible placements for 2D tensor
        ALL_PLACEMENTS = [
            Replicate(),
            Shard(0),
            Shard(1),
            Partial("sum"),
            Partial("max"),
        ]

        # Test each operator
        for op, valid_rule_strs in VALID_RULES.items():
            valid_rules = {parse_rule(r) for r in valid_rule_strs}

            # Create test tensors
            a = torch.randn(8, 4)
            b = torch.randn(8, 4)
            sample = SampleInput(a, args=(b,))
            tensors = extract_tensors_from_sample(sample)
            ground_truth = op(a, b)

            with LocalTensorMode(frozenset(range(self.world_size))):
                mesh = init_device_mesh("cpu", (self.world_size,))

                # Test all combinations
                for p1 in ALL_PLACEMENTS:
                    for p2 in ALL_PLACEMENTS:
                        # Skip fully replicated inputs (degenerate case: any output works)
                        if isinstance(p1, Replicate) and isinstance(p2, Replicate):
                            continue

                        for p_out in ALL_PLACEMENTS:
                            input_plcs = (p1, p2)
                            combo = PlacementCombination(input_plcs, p_out)

                            is_valid, msg = validate_combination(
                                op,
                                sample,
                                tensors,
                                combo,
                                ground_truth,
                                self.world_size,
                                mesh,
                            )

                            # Check if this combo matches any valid rule
                            should_be_valid = (input_plcs, p_out) in valid_rules

                            if should_be_valid:
                                self.assertTrue(
                                    is_valid,
                                    f"{op.__name__}: {p1},{p2}->{p_out} should be valid but got: {msg}",
                                )
                            else:
                                self.assertFalse(
                                    is_valid,
                                    f"{op.__name__}: {p1},{p2}->{p_out} should be invalid",
                                )

    def test_add_alpha_negates_partial_max_to_min(self):
        """
        Test that add with alpha=-1 converts P(max) to P(min).

        When computing a + alpha*b with alpha=-1:
        - If b is P(max), then -b behaves as P(min)
        - So R + (-1)*P(max) -> P(min) should be valid
        - And R + (-1)*P(max) -> P(max) should be invalid
        """
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        # add with alpha=-1 computes: a + (-1)*b = a - b
        sample = SampleInput(a, args=(b,), kwargs={"alpha": -1})
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.add(a, b, alpha=-1)

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))

            # R + alpha*P(max) where alpha=-1 should produce P(min)
            # because -max(x) = min(-x)
            combo_valid = PlacementCombination(
                input_placements=(Replicate(), Partial("max")),
                output_placement=Partial("min"),
            )
            is_valid, msg = validate_combination(
                torch.add,
                sample,
                tensors,
                combo_valid,
                ground_truth,
                self.world_size,
                mesh,
            )
            self.assertTrue(
                is_valid, f"R,Pmax->Pmin with alpha=-1 should be valid: {msg}"
            )

            # R + alpha*P(max) where alpha=-1 should NOT produce P(max)
            combo_invalid = PlacementCombination(
                input_placements=(Replicate(), Partial("max")),
                output_placement=Partial("max"),
            )
            is_valid, msg = validate_combination(
                torch.add,
                sample,
                tensors,
                combo_invalid,
                ground_truth,
                self.world_size,
                mesh,
            )
            self.assertFalse(is_valid, "R,Pmax->Pmax with alpha=-1 should be invalid")

            # Similarly, P(max) + alpha*R where alpha=-1 should produce P(max)
            # because we're subtracting a replicated value from P(max)
            combo_pmax_minus_r = PlacementCombination(
                input_placements=(Partial("max"), Replicate()),
                output_placement=Partial("max"),
            )
            is_valid, msg = validate_combination(
                torch.add,
                sample,
                tensors,
                combo_pmax_minus_r,
                ground_truth,
                self.world_size,
                mesh,
            )
            self.assertTrue(
                is_valid, f"Pmax,R->Pmax with alpha=-1 should be valid: {msg}"
            )


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

    def test_tensor_idx_produces_different_sum_ratios(self):
        """
        Verify that different tensor_idx values produce different P(sum) ratios.

        This is critical for catching invalid rules like mul P(sum),P(sum)->P(sum).
        If both tensors used the same 0.5/0.5 split, some invalid combinations
        might accidentally produce correct results.
        """
        tensor = torch.ones(4, 3)

        # Create P(sum) with different tensor_idx values
        local0 = _create_partial_input(
            tensor, Partial("sum"), world_size=2, tensor_idx=0
        )
        local1 = _create_partial_input(
            tensor, Partial("sum"), world_size=2, tensor_idx=1
        )

        # Both should reduce to the same original tensor
        self.assertTrue(
            torch.allclose(local0._local_tensors[0] + local0._local_tensors[1], tensor)
        )
        self.assertTrue(
            torch.allclose(local1._local_tensors[0] + local1._local_tensors[1], tensor)
        )

        # But the per-rank values should be DIFFERENT
        # (tensor_idx=0 uses ratio 0.6, tensor_idx=1 uses ratio 0.7)
        self.assertFalse(
            torch.allclose(local0._local_tensors[0], local1._local_tensors[0]),
            "Different tensor_idx should produce different P(sum) ratios",
        )

    def test_tensor_idx_produces_different_max_patterns(self):
        """
        Verify that different tensor_idx values produce different P(max) offset patterns.

        This is critical for catching invalid rules like add P(max),P(max)->P(max).
        If both tensors used the same offset pattern, the offsets would cancel out
        and invalid combinations might appear valid.
        """
        tensor = torch.zeros(4, 3)

        # Create P(max) with different tensor_idx values
        local0 = _create_partial_input(
            tensor, Partial("max"), world_size=2, tensor_idx=0
        )
        local1 = _create_partial_input(
            tensor, Partial("max"), world_size=2, tensor_idx=1
        )

        # Both should reduce to the same original tensor
        stacked0 = torch.stack([local0._local_tensors[r] for r in range(2)])
        stacked1 = torch.stack([local1._local_tensors[r] for r in range(2)])
        self.assertTrue(torch.allclose(stacked0.max(dim=0).values, tensor))
        self.assertTrue(torch.allclose(stacked1.max(dim=0).values, tensor))

        # The offset patterns should be different (shifted by 1 element)
        # tensor_idx=0: rank0 has [0, -1.3, 0, -1.3, ...], rank1 has [-1.3, 0, ...]
        # tensor_idx=1: rank0 has [-1.3, 0, -1.3, 0, ...], rank1 has [0, -1.3, ...]
        self.assertFalse(
            torch.allclose(local0._local_tensors[0], local1._local_tensors[0]),
            "Different tensor_idx should produce different P(max) patterns",
        )


class TestPartialCombinationValidity(TestCase):
    """
    Tests that verify invalid partial combinations are correctly detected.

    These tests would FAIL if the partial creation logic were simplified
    (e.g., using symmetric splits or ignoring tensor_idx).
    """

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_mul_psum_psum_is_invalid(self):
        """
        Verify that mul with P(sum),P(sum)->P(sum) is detected as INVALID.

        Mathematically: (a0+a1) * (b0+b1) != (a0*b0) + (a1*b1)
        This test would FAIL if P(sum) used symmetric 0.5/0.5 splits for both
        tensors, because then a0=a1 and b0=b1, making:
        (0.5a+0.5a) * (0.5b+0.5b) = a*b = (0.5a*0.5b)*4 which could accidentally match.
        """
        a = torch.randn(8, 4) + 2  # Offset to avoid near-zero values
        b = torch.randn(8, 4) + 2
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a * b

        combo = PlacementCombination(
            input_placements=(Partial("sum"), Partial("sum")),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.mul, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertFalse(
            is_valid,
            "mul Psum,Psum->Psum should be invalid (multiplication doesn't distribute over addition)",
        )

    def test_add_pmax_pmax_is_invalid(self):
        """
        Verify that add with P(max),P(max)->P(max) is detected as INVALID.

        When two P(max) tensors with different offset patterns are added,
        the negative offsets can accumulate, causing the max to not equal
        the ground truth.

        This test would FAIL if both P(max) inputs used the same offset pattern,
        because then the offsets would align and potentially cancel correctly.
        """
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Partial("max"), Partial("max")),
            output_placement=Partial("max"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertFalse(
            is_valid,
            "add Pmax,Pmax->Pmax should be invalid (offsets accumulate with different patterns)",
        )

    def test_mixed_partial_types_invalid(self):
        """
        Verify that mixing different partial types is detected as invalid.

        P(sum) + P(max) -> anything should always be invalid because
        the reduction semantics are incompatible.
        """
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        # Try all output placements - none should be valid
        for out_reduce_op in ["sum", "max", "min", "avg"]:
            combo = PlacementCombination(
                input_placements=(Partial("sum"), Partial("max")),
                output_placement=Partial(out_reduce_op),
            )

            with LocalTensorMode(frozenset(range(self.world_size))):
                mesh = init_device_mesh("cpu", (self.world_size,))
                is_valid, msg = validate_combination(
                    torch.add,
                    sample,
                    tensors,
                    combo,
                    ground_truth,
                    self.world_size,
                    mesh,
                )

            self.assertFalse(
                is_valid,
                f"add Psum,Pmax->P({out_reduce_op}) should be invalid",
            )

    def test_pmin_with_replicate_valid(self):
        """
        Verify that P(min) + R -> P(min) is valid for add.

        This mirrors the P(max) + R -> P(max) case but for min.
        """
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a + b

        combo = PlacementCombination(
            input_placements=(Partial("min"), Replicate()),
            output_placement=Partial("min"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.add, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertTrue(is_valid, f"add Pmin,R->Pmin should be valid: {msg}")

    def test_all_zero_output_false_positive(self):
        """
        When ground truth is all zeros, every placement trivially validates.

        Zeros are a fixed point of all reduce operations (sum, max, min),
        so any Partial output placement matches ground truth. This means
        invalid rules like mul P(sum),P(sum)->P(sum) appear valid when
        inputs produce all-zero output.
        """
        a = torch.zeros(8, 4)
        b = torch.randn(8, 4)
        sample = SampleInput(a, args=(b,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = a * b  # all zeros

        # P(sum),P(sum)->P(sum) is NOT valid for mul in general, but
        # with all-zero output it trivially passes: sum(0,0)=0=ground_truth
        combo = PlacementCombination(
            input_placements=(Partial("sum"), Partial("sum")),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.mul, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertFalse(
            is_valid,
            "mul Psum,Psum->Psum should be invalid even with all-zero output",
        )


class TestQuerySingleDimStrategyKwargs(TestCase):
    """Test that query_single_dim_strategy forwards kwargs to strategy functions."""

    def test_kwargs_forwarded_to_strategy(self):
        """
        A kwargs-aware strategy should receive the actual kwargs, not {}.

        torch.add(a, b, alpha=-1) changes which Partial rules are valid:
        with alpha=1, R,P(max)->P(max) is valid; with alpha=-1, it becomes
        R,P(max)->P(min) instead. A strategy that accounts for alpha needs
        to receive it through kwargs.
        """
        from torch.distributed.tensor._api import DTensor
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
        )

        propagator = DTensor._op_dispatcher.sharding_propagator
        aten_add = torch.ops.aten.add.Tensor

        # A strategy that returns different rules depending on alpha.
        # With alpha >= 0: R,P(max)->P(max) is valid
        # With alpha < 0:  R,P(max)->P(min) is valid (negation flips max to min)
        def alpha_aware_add_strategy(op, args_schema, kwargs_schema):
            alpha = kwargs_schema.get("alpha", 1)
            rules = [
                [
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                ],
                [Partial("sum"), Partial("sum"), Partial("sum")],
            ]
            if alpha < 0:
                rules.append([Partial("min"), Replicate(), Partial("max")])
            else:
                rules.append([Partial("max"), Replicate(), Partial("max")])
            return rules

        original = propagator.op_single_dim_strategy_funcs.get(aten_add)
        propagator.op_single_dim_strategy_funcs[aten_add] = alpha_aware_add_strategy
        try:
            tensors = [("a", torch.randn(4, 3)), ("b", torch.randn(4, 3))]

            # Query with alpha=-1 kwargs
            result = query_single_dim_strategy(
                aten_add, tensors, None, kwargs={"alpha": -1}
            )
            self.assertIsNotNone(result)

            # The third rule's output should be P(min) for alpha=-1
            self.assertEqual(len(result), 3)
            self.assertIsInstance(result[2][0], Partial)
            self.assertEqual(
                result[2][0].reduce_op,
                "min",
                "With alpha=-1, the strategy should produce P(min) output "
                "but got P(max) — kwargs were not forwarded",
            )
        finally:
            if original is not None:
                propagator.op_single_dim_strategy_funcs[aten_add] = original
            else:
                propagator.op_single_dim_strategy_funcs.pop(aten_add, None)


if __name__ == "__main__":
    run_tests()
