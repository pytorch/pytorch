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
    _checkerboard_mask,
    _create_partial_input,
    _find_opinfo_candidates,
    extract_tensors_from_sample,
    get_1d_input_placements_for_tensor,
    get_1d_output_placements_for_tensor,
    get_opinfo_by_name,
    is_fully_replicated,
    is_trivial_shard,
    normalize_combo_key,
    normalize_placement,
    normalize_placement_str,
    parse_placement,
    placement_tuple_to_str,
    PlacementCombination,
    query_single_dim_strategy,
    resolve_op_names,
    validate_combination,
)
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.testing._internal.common_utils import run_tests, TEST_WITH_SLOW, TestCase


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

    def test_nan_output_valid_when_both_nan(self):
        """Test that NaN outputs are considered valid when both ground truth and sharded match.

        igamma with negative inputs produces NaN. When sharding produces the same
        NaN pattern, the rule should be valid (NaN == NaN for validation purposes).
        """
        # igamma(negative, negative) produces NaN
        a = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
        x = torch.tensor(-1.0)
        sample = SampleInput(a, args=(x,))
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.igamma(a, x)
        self.assertTrue(ground_truth.isnan().all(), "Expected all NaN ground truth")

        combo = PlacementCombination(
            input_placements=(Shard(0), Replicate()), output_placement=Shard(0)
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.igamma,
                sample,
                tensors,
                combo,
                ground_truth,
                self.world_size,
                mesh,
            )
        self.assertTrue(is_valid, f"NaN outputs should match: {msg}")

    def test_integer_output_includes_partial_sum(self):
        """
        P(sum) must be included in output placements for integer dtypes.

        Ops like bucketize use R, S(0) -> P(sum) where the output is integer
        (bucket indices). If P(sum) is excluded from the validator's output
        enumeration for integer types, these rules appear as false "incorrect"
        reports because the ground truth enumeration never discovers them.
        """
        for dtype in [torch.int32, torch.int64]:
            t = torch.zeros(4, dtype=dtype)
            placements = get_1d_output_placements_for_tensor(t)
            partial_ops = {p.reduce_op for p in placements if isinstance(p, Partial)}
            self.assertIn(
                "sum",
                partial_ops,
                f"P(sum) must be in output placements for {dtype} "
                f"(needed by ops like bucketize with sharded boundaries)",
            )

    def test_exhaustive_binary_op_rules(self):
        """
        Exhaustively test all placement combinations for binary ops.

        For each op, we define the complete set of valid rules. The test then:
        1. Verifies all listed rules are detected as valid
        2. Verifies all unlisted combinations are detected as invalid
        """

        def parse_rule(rule_str):
            """Parse 'S(0),S(0)->S(0)' into ((Shard(0), Shard(0)), Shard(0))."""
            inputs_str, output_str = rule_str.split("->")
            inputs = tuple(parse_placement(s.strip()) for s in inputs_str.split(","))
            return (inputs, parse_placement(output_str.strip()))

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
        # But NOT:
        # - R + P(sum) -> P(sum) for add (R gets added on each rank, then summed)
        VALID_RULES = {
            torch.add: [
                # Same-dim sharding (chunks match)
                "S(0),S(0)->S(0)",
                "S(1),S(1)->S(1)",
                # Partial sum/avg + Partial sum/avg -> same (linearity of addition)
                # (a0+a1) + (b0+b1) = (a0+b0) + (a1+b1) where ai/bi are per-rank
                "P(sum),P(sum)->P(sum)",
                "P(avg),P(avg)->P(avg)",
                # Partial sum/avg with Replicate: avg normalizes the extra
                # copies, so P(avg)+R works even though P(sum)+R does not
                "P(avg),R->P(avg)",
                "R,P(avg)->P(avg)",
                # Partial max/min with Replicate: adding a constant preserves
                # the reduce structure (NOT Pmax+Pmax, offsets accumulate)
                "P(max),R->P(max)",
                "P(min),R->P(min)",
                # NOTE: these two rules are NOT valid in general for torch.add since it accepts alpha=a, which if negative
                # flips the the partial output from max to min or vice versa.
                # However, this test is simpler than the end to end validator and ignores alpha, and the rules have to
                # be listed as valid since without alpha they DO produce correct results and the test asserts any rule
                # NOT listed here produces incorrect results.
                "R,P(max)->P(max)",
                "R,P(min)->P(min)",
            ],
            torch.mul: [
                # Same-dim sharding
                "S(0),S(0)->S(0)",
                "S(1),S(1)->S(1)",
                # Partial sum/avg * Replicate -> same (multiplicative linearity)
                # r * (p0+p1) = r*p0 + r*p1 where pi are per-rank
                "P(sum),R->P(sum)",
                "R,P(sum)->P(sum)",
                "P(avg),R->P(avg)",
                "R,P(avg)->P(avg)",
                # No P(min)/P(max) rules: negative multiplier flips ordering
            ],
            torch.div: [
                # Same-dim sharding
                "S(0),S(0)->S(0)",
                "S(1),S(1)->S(1)",
                # Partial sum/avg / Replicate -> same (division by constant is linear)
                "P(sum),R->P(sum)",
                "P(avg),R->P(avg)",
                # No R/P(avg) rule: 1/x is not linear
                # No P(min)/P(max) rules: negative divisor flips ordering
            ],
            torch.maximum: [
                # Same-dim sharding
                "S(0),S(0)->S(0)",
                "S(1),S(1)->S(1)",
                # Partial max + Partial max -> Partial max
                # max(max(a0,a1), max(b0,b1)) = max(max(a0,b0), max(a1,b1))
                "P(max),P(max)->P(max)",
                # Partial max/min with Replicate (lattice distributivity):
                # max(min(a0,a1), r) = min(max(a0,r), max(a1,r)) ✓
                "P(max),R->P(max)",
                "R,P(max)->P(max)",
                "P(min),R->P(min)",
                "R,P(min)->P(min)",
                # No P(avg) rules: max is not linear
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
        if TEST_WITH_SLOW:
            # This makes the test go from 4 sec to 12 sec, and I don't think it really adds much useful coverage, but
            # why not have it run in CI.
            ALL_PLACEMENTS += [Partial("avg"), Partial("min")]

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

        # The offset patterns should be different (shifted checkerboard)
        self.assertFalse(
            torch.allclose(local0._local_tensors[0], local1._local_tensors[0]),
            "Different tensor_idx should produce different P(max) patterns",
        )

    def test_checkerboard_mask_alternates_every_dim(self):
        """
        The checkerboard mask must alternate along every dimension.

        A flat-index mask (idx % 2) has uniform parity along even-stride
        dimensions: for a [3,2,1,2] tensor, dim 0 stride is 4, so all
        elements along dim 0 at a given position share the same flat-index
        parity. This causes argmin/argmax to produce identical results on
        all ranks, making P(min)/P(max) inputs falsely appear valid.

        The checkerboard mask (sum-of-coordinates % 2) avoids this by
        guaranteeing adjacent elements differ along every axis.
        """
        # Shape where flat-index mask fails: stride of dim 0 = 4 (even)
        tensor = torch.randn(3, 2, 1, 2)
        mask = _checkerboard_mask(tensor)

        # Reshape mask back to tensor shape to check along each dim
        mask_nd = mask.reshape(tensor.shape)

        # Along dim 0, adjacent elements must alternate
        for j in range(2):
            for k in range(1):
                for l in range(2):
                    vals = [mask_nd[i, j, k, l].item() for i in range(3)]
                    # Check alternation: each consecutive pair should differ
                    for idx in range(len(vals) - 1):
                        self.assertNotEqual(
                            vals[idx],
                            vals[idx + 1],
                            f"Mask should alternate along dim 0 at [{':,'}{j},{k},{l}], "
                            f"got {vals}",
                        )

    def test_pmin_pmax_adaptive_offset_exceeds_range(self):
        """
        P(min)/P(max) offsets must exceed the tensor's value range.

        With fixed small offsets (e.g., 0.7 / -1.3), the true minimum
        dominates across all ranks, making argmin produce the same index
        everywhere. Adaptive offsets scaled to 2*range+1 ensure the mask
        pattern determines which rank finds the argmin, so different ranks
        disagree for index-returning ops.
        """
        # Tensor with large value range — fixed offsets would be too small
        tensor = torch.tensor([-10.0, -1.0, 5.0, 8.0])
        value_range = (tensor.max() - tensor.min()).item()  # 18.0

        for reduce_op, sign in [("min", 1), ("max", -1)]:
            local = _create_partial_input(tensor, Partial(reduce_op), world_size=2)
            r0, r1 = local._local_tensors[0], local._local_tensors[1]
            # The actual offset applied should exceed value_range
            max_offset = (r0 - r1).abs().max().item()
            self.assertGreater(
                max_offset,
                value_range,
                f"P({reduce_op}) offset {max_offset:.1f} should exceed "
                f"value range {value_range:.1f}",
            )
            # Ranks should disagree on argmin/argmax
            self.assertNotEqual(
                r0.argmin().item(),
                r1.argmin().item(),
                f"P({reduce_op}) ranks should disagree on argmin with adaptive offset",
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

    def test_abs_psum_psum_is_invalid(self):
        """
        P(sum)->P(sum) is NOT valid for abs (or any non-linear elementwise op).

        abs(a0 + a1) != abs(a0) + abs(a1) when a0 and a1 have different signs.
        The _create_partial_input function must create local values with mixed
        signs to expose this, not just proportional splits (tensor * ratio)
        that preserve the sign pattern.
        """
        t = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        sample = SampleInput(t)
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.abs(t)

        combo = PlacementCombination(
            input_placements=(Partial("sum"),),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.abs, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertFalse(
            is_valid,
            "abs P(sum)->P(sum) should be invalid (abs is non-linear)",
        )

    def test_abs_pavg_pavg_is_invalid(self):
        """P(avg)->P(avg) is NOT valid for abs, same reasoning as P(sum)."""
        t = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        sample = SampleInput(t)
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.abs(t)

        combo = PlacementCombination(
            input_placements=(Partial("avg"),),
            output_placement=Partial("avg"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.abs, sample, tensors, combo, ground_truth, self.world_size, mesh
            )

        self.assertFalse(
            is_valid,
            "abs P(avg)->P(avg) should be invalid (abs is non-linear)",
        )

    def test_all_zero_output_false_positive(self):
        """
        All-zero ground truth makes every placement trivially validate.

        Zeros are a fixed point of all reduce operations (sum, max, min),
        so validate_combination cannot distinguish valid from invalid rules.
        This is a known limitation: compare_operator skips all-zero samples
        to avoid hundreds of false positive rules.

        We use zeros_like as the test op because it always produces zeros
        regardless of input values, unlike mul(zeros, x) which the offset
        fix in _create_partial_input now correctly handles.
        """
        t = torch.randn(8, 4)
        sample = SampleInput(t)
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.zeros_like(t)

        # P(sum)->P(sum) is NOT valid for zeros_like, but with all-zero
        # output it trivially passes: sum(0,0)=0=ground_truth.
        combo = PlacementCombination(
            input_placements=(Partial("sum"),),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.zeros_like,
                sample,
                tensors,
                combo,
                ground_truth,
                self.world_size,
                mesh,
            )

        # This demonstrates the false positive: validate_combination says
        # valid, even though the rule is invalid for non-zero inputs.
        self.assertTrue(
            is_valid,
            "Expected True (false positive) for all-zero output, showing "
            "why compare_operator must skip such samples",
        )

    def test_argmin_rejects_pmax_input(self):
        """
        P(max) -> R must be rejected for argmin.

        argmin returns indices, not values. With P(max) input, different
        ranks hold the true value at different positions (checkerboard mask),
        so argmin produces different indices on each rank. The Replicate
        output check should catch this disagreement.

        This test exercises both fixes:
        1. Adaptive offset (exceeds value range) — prevents the true min
           from dominating on all ranks despite the perturbation
        2. Checkerboard mask — prevents uniform parity along even-stride
           dimensions, which would make all elements in a reduction group
           shift identically
        """
        # Use a shape where flat-index mask would fail: dim 0 stride = 4 (even)
        t = torch.randn(3, 2, 1, 2)
        sample = SampleInput(t, kwargs={"dim": 0, "keepdim": True})
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.argmin(t, dim=0, keepdim=True)

        for reduce_op in ("min", "max"):
            combo = PlacementCombination(
                input_placements=(Partial(reduce_op),),
                output_placement=Replicate(),
            )
            with LocalTensorMode(frozenset(range(self.world_size))):
                mesh = init_device_mesh("cpu", (self.world_size,))
                is_valid, msg = validate_combination(
                    torch.argmin,
                    sample,
                    tensors,
                    combo,
                    ground_truth,
                    self.world_size,
                    mesh,
                )
            self.assertFalse(
                is_valid,
                f"argmin P({reduce_op})->R should be invalid "
                f"(index op, ranks disagree): {msg}",
            )
        """
        All-zero ground truth makes every placement trivially validate.

        Zeros are a fixed point of all reduce operations (sum, max, min),
        so validate_combination cannot distinguish valid from invalid rules.
        This is a known limitation: compare_operator skips all-zero samples
        to avoid hundreds of false positive rules.

        We use zeros_like as the test op because it always produces zeros
        regardless of input values, unlike mul(zeros, x) which the offset
        fix in _create_partial_input now correctly handles.
        """
        t = torch.randn(8, 4)
        sample = SampleInput(t)
        tensors = extract_tensors_from_sample(sample)
        ground_truth = torch.zeros_like(t)

        # P(sum)->P(sum) is NOT valid for zeros_like, but with all-zero
        # output it trivially passes: sum(0,0)=0=ground_truth.
        combo = PlacementCombination(
            input_placements=(Partial("sum"),),
            output_placement=Partial("sum"),
        )

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            is_valid, msg = validate_combination(
                torch.zeros_like,
                sample,
                tensors,
                combo,
                ground_truth,
                self.world_size,
                mesh,
            )

        # This demonstrates the false positive: validate_combination says
        # valid, even though the rule is invalid for non-zero inputs.
        self.assertTrue(
            is_valid,
            "Expected True (false positive) for all-zero output, showing "
            "why compare_operator must skip such samples",
        )


class TestDecompStrategyPath(TestCase):
    """Test that decomposition-based strategy propagation discovers rules."""

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_decomp_discovers_rules_for_softplus(self):
        """
        Exercise DecompShardingStrategy.propagate_strategy directly on a unary
        op (softplus) whose decomposition uses only registered ops.

        Verifies the helper _extract_rules_from_op_strategy correctly extracts
        elementwise sharding rules from the resulting OpStrategy.
        """
        try:
            from torch.distributed.tensor._decompositions import DecompShardingStrategy
        except ImportError:
            self.skipTest("_decompositions module not available")
        from torch.distributed.tensor._dtensor_spec import TensorMeta
        from torch.distributed.tensor._op_schema import DTensorSpec, OpSchema
        from torch.distributed.tensor._ops.strategy_validation import (
            _extract_rules_from_op_strategy,
        )

        aten_softplus = torch.ops.aten.softplus.default
        propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        self.assertTrue(DecompShardingStrategy.has_decomp(aten_softplus))

        t = torch.randn(8, 4)

        mesh = init_device_mesh("cpu", (self.world_size,))
        spec = DTensorSpec(
            mesh=mesh,
            placements=(Shard(0),),
            tensor_meta=TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype),
        )
        op_schema = OpSchema(aten_softplus, (spec,), {})
        propagator.decomp_strategy.ensure_schema_info(aten_softplus)
        output_strategy = propagator.decomp_strategy.propagate_strategy(
            op_schema,
        )
        self.assertIsNotNone(output_strategy)

        input_shapes = (t.shape,)
        output_shape = tuple(torch.nn.functional.softplus(t).shape)
        rules = _extract_rules_from_op_strategy(
            output_strategy, input_shapes, output_shape
        )

        # Should discover elementwise sharding rules for a 2D tensor
        self.assertIn((("S(0)",), "S(0)"), rules)
        self.assertIn((("S(1)",), "S(1)"), rules)

    def test_compare_operator_uses_decomp_path(self):
        """
        compare_operator should find rules via the decomp path when an op is
        not registered in op_strategy_funcs or op_single_dim_strategy_funcs.

        Temporarily removes addcmul from strategy registries to force the
        decomp fallback, and uses even tensor sizes for correct sharding.
        """
        try:
            from torch.distributed.tensor._decompositions import (  # noqa: F401
                DecompShardingStrategy,
            )
        except ImportError:
            self.skipTest("_decompositions module not available")

        import torch.testing._internal.common_methods_invocations as common_ops
        from torch.distributed.tensor._ops.strategy_validation import compare_operator
        from torch.testing._internal.opinfo import core as opinfo_core

        propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        aten_addcmul = torch.ops.aten.addcmul.default

        # Save and remove addcmul from strategy registries to force decomp path
        saved_strategy = propagator.op_strategy_funcs.pop(aten_addcmul, None)
        saved_single = propagator.op_single_dim_strategy_funcs.pop(aten_addcmul, None)

        # Override sizes to ensure even sharding with world_size=2
        orig_sizes = (opinfo_core.L, opinfo_core.M, opinfo_core.S, opinfo_core.XS)
        opinfo_core.L = common_ops.L = 24
        opinfo_core.M = common_ops.M = 12
        opinfo_core.S = common_ops.S = 4
        opinfo_core.XS = common_ops.XS = 2

        try:
            # compare_operator manages its own process group
            stats = compare_operator(
                "addcmul",
                device="cpu",
                dtype=torch.float32,
                world_size=self.world_size,
                incorrect_only=True,
            )
            # Decomp should discover valid rules
            self.assertGreater(stats.true_positives, 0)
            # No incorrect rules
            self.assertEqual(len(stats.false_positives), 0)
        finally:
            # Restore registries and sizes
            if saved_strategy is not None:
                propagator.op_strategy_funcs[aten_addcmul] = saved_strategy
            if saved_single is not None:
                propagator.op_single_dim_strategy_funcs[aten_addcmul] = saved_single
            (
                opinfo_core.L,
                opinfo_core.M,
                opinfo_core.S,
                opinfo_core.XS,
            ) = orig_sizes
            (
                common_ops.L,
                common_ops.M,
                common_ops.S,
                common_ops.XS,
            ) = orig_sizes


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


class TestCompareOperatorEndToEnd(TestCase):
    """End-to-end smoke test for compare_operator."""

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _with_even_sizes(self, fn):
        """Run fn with opinfo sizes overridden to be evenly divisible by world_size."""
        import torch.testing._internal.common_methods_invocations as common_ops
        from torch.testing._internal.opinfo import core as opinfo_core

        orig_sizes = (opinfo_core.L, opinfo_core.M, opinfo_core.S, opinfo_core.XS)
        opinfo_core.L = common_ops.L = 24
        opinfo_core.M = common_ops.M = 12
        opinfo_core.S = common_ops.S = 4
        opinfo_core.XS = common_ops.XS = 2
        try:
            return fn()
        finally:
            (
                opinfo_core.L,
                opinfo_core.M,
                opinfo_core.S,
                opinfo_core.XS,
            ) = orig_sizes
            (
                common_ops.L,
                common_ops.M,
                common_ops.S,
                common_ops.XS,
            ) = orig_sizes

    def test_compare_operator_add_no_incorrect(self):
        """compare_operator on add should find valid rules with no incorrect ones."""
        from torch.distributed.tensor._ops.strategy_validation import compare_operator

        def run():
            stats = compare_operator(
                "add",
                device="cpu",
                dtype=torch.float32,
                world_size=self.world_size,
                incorrect_only=True,
            )
            self.assertGreater(stats.true_positives, 0)
            self.assertEqual(len(stats.false_positives), 0)

        self._with_even_sizes(run)

    def test_compare_operator_split_multi_output(self):
        """compare_operator should handle multi-output ops like split."""
        from torch.distributed.tensor._ops.strategy_validation import compare_operator

        def run():
            stats = compare_operator(
                "split",
                device="cpu",
                dtype=torch.float32,
                world_size=self.world_size,
                incorrect_only=True,
            )
            self.assertGreater(
                stats.total_samples,
                0,
                f"split should have runnable samples, got skip_reasons={stats.skip_reasons}",
            )
            self.assertEqual(len(stats.false_positives), 0)

        self._with_even_sizes(run)


class TestMainModule(TestCase):
    """Test that strategy_validation can be run as a main module."""

    def _run_module(self, *extra_args):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.tensor._ops.strategy_validation",
                "--device",
                "cpu",
                *extra_args,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Module exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[-2000:]}",
        )
        return result.stdout

    def test_run_as_module_default(self):
        """Running with no args should validate 'add' and exit cleanly."""
        stdout = self._run_module()
        self.assertIn("add", stdout)
        self.assertIn("Correct", stdout)

    def test_run_with_op_flag(self):
        """Running with --op mul --incorrect-only should work."""
        stdout = self._run_module("--op", "mul", "--incorrect-only")
        self.assertIn("mul", stdout)


class TestOpInfoLookup(TestCase):
    """Tests for get_opinfo_by_name, _find_opinfo_candidates, and resolve_op_names."""

    def test_get_opinfo_by_name_exact(self):
        results = get_opinfo_by_name("add")
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertEqual(op.name, "add")

    def test_get_opinfo_by_name_qualified(self):
        results = get_opinfo_by_name("nn.functional.relu")
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertEqual(op.name, "nn.functional.relu")

    def test_get_opinfo_by_name_not_found_with_suggestions(self):
        with self.assertRaises(ValueError) as ctx:
            get_opinfo_by_name("relu")
        self.assertIn("did you mean", str(ctx.exception))
        self.assertIn("nn.functional.relu", str(ctx.exception))

    def test_get_opinfo_by_name_not_found_no_suggestions(self):
        with self.assertRaises(ValueError) as ctx:
            get_opinfo_by_name("this_op_does_not_exist_xyz")
        self.assertNotIn("did you mean", str(ctx.exception))

    def test_find_opinfo_candidates_aten_name(self):
        # relu OpInfo has aten_name="relu" explicitly set
        candidates = _find_opinfo_candidates("relu")
        self.assertIn("nn.functional.relu", candidates)

    def test_find_opinfo_candidates_suffix(self):
        candidates = _find_opinfo_candidates("dropout")
        self.assertIn("nn.functional.dropout", candidates)

    def test_find_opinfo_candidates_no_match(self):
        candidates = _find_opinfo_candidates("this_op_does_not_exist_xyz")
        self.assertEqual(candidates, [])

    def test_resolve_op_names_exact(self):
        result = resolve_op_names(["add"])
        self.assertEqual(result, ["add"])

    def test_resolve_op_names_qualified(self):
        result = resolve_op_names(["nn.functional.relu"])
        self.assertEqual(result, ["nn.functional.relu"])

    def test_resolve_op_names_multiple(self):
        result = resolve_op_names(["add", "mul"])
        self.assertEqual(result, ["add", "mul"])

    def test_resolve_op_names_deduplicates(self):
        result = resolve_op_names(["add", "add"])
        self.assertEqual(result, ["add"])

    def test_resolve_op_names_glob(self):
        result = resolve_op_names(["nn.functional.relu*"])
        self.assertIn("nn.functional.relu", result)

    def test_resolve_op_names_glob_no_match(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_op_names(["zzz_no_match_*"])
        self.assertIn("No OpInfo names match", str(ctx.exception))

    def test_resolve_op_names_unambiguous_shorthand(self):
        # "relu" should resolve unambiguously to "nn.functional.relu"
        result = resolve_op_names(["nn.functional.relu"])
        self.assertIn("nn.functional.relu", result)

    def test_resolve_op_names_ambiguous_shorthand(self):
        # "dropout" matches nn.functional.dropout, nn.functional.dropout2d, etc.
        candidates = _find_opinfo_candidates("dropout")
        if len(candidates) > 1:
            with self.assertRaises(ValueError) as ctx:
                resolve_op_names(["dropout"])
            self.assertIn("ambiguous", str(ctx.exception))

    def test_resolve_op_names_not_found(self):
        with self.assertRaises(ValueError):
            resolve_op_names(["this_op_does_not_exist_xyz"])


if __name__ == "__main__":
    run_tests()
