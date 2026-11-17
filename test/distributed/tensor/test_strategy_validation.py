# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for the strategy validation library.

These tests verify that the validation logic correctly detects:
- Incorrect rules (DTensor claims valid but produces wrong results)
- Missing rules (ground truth valid but DTensor has no rule)
"""

import itertools
import math
import unittest.mock

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed._local_tensor._c10d import _local_reduce
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor._ops.single_dim_strategy import (
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.strategy_validation import (
    _checkerboard_mask,
    _compare_outputs,
    _constrained_operands,
    _find_opinfo_candidates,
    _inject_specials,
    _materialize_inputs,
    _mismatch,
    _opinfo_sizes,
    _partial_pieces,
    _scalar_variants,
    _validate_aten_op_for_sample,
    _validate_captured_call,
    _Variant,
    claimed_input_combinations,
    ComparisonStats,
    extract_tensors_from_sample,
    get_1d_input_placements_for_tensor,
    get_1d_output_placements_for_tensor,
    get_opinfo_by_name,
    is_fully_replicated,
    is_trivial_shard,
    parse_placement,
    query_runtime_claims,
    resolve_op_names,
    validate_aten_combination,
    ValidationOptions,
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

    def test_is_fully_replicated(self):
        self.assertTrue(is_fully_replicated((Replicate(),)))
        self.assertTrue(is_fully_replicated((Replicate(), Replicate())))
        self.assertFalse(is_fully_replicated((Shard(0),)))
        self.assertFalse(is_fully_replicated((Replicate(), Shard(0))))
        self.assertFalse(is_fully_replicated((Partial("sum"),)))


class TestTrivialShard(TestCase):
    """Test detection of shards on size-1 dimensions."""

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
    """Test validate_aten_combination on hand-picked placement combinations."""

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
        self.assertTrue(torch.igamma(a, x).isnan().all(), "Expected all NaN output")

        combo = ((Shard(0), Replicate()), (Shard(0),))

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
        is_valid, msg = validate_aten_combination(
            torch.ops.aten.igamma.default, (a, x), {}, combo, self.world_size, mesh
        )
        self.assertTrue(is_valid, lambda _m: f"{_m}\nNaN outputs should match: {msg}")

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
                lambda msg: f"{msg}\nP(sum) must be in output placements for {dtype} "
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
            return (inputs, (parse_placement(output_str.strip()),))

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
            torch.ops.aten.add.Tensor: [
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
                # flips the partial output from max to min or vice versa.
                # However, this test is simpler than the end to end validator and ignores alpha, and the rules have to
                # be listed as valid since without alpha they DO produce correct results and the test asserts any rule
                # NOT listed here produces incorrect results.
                "R,P(max)->P(max)",
                "R,P(min)->P(min)",
            ],
            torch.ops.aten.mul.Tensor: [
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
            torch.ops.aten.div.Tensor: [
                # Same-dim sharding
                "S(0),S(0)->S(0)",
                "S(1),S(1)->S(1)",
                # Partial sum/avg / Replicate -> same (division by constant is linear)
                "P(sum),R->P(sum)",
                "P(avg),R->P(avg)",
                # No R/P(avg) rule: 1/x is not linear
                # No P(min)/P(max) rules: negative divisor flips ordering
            ],
            torch.ops.aten.maximum.default: [
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
                        combo = (input_plcs, (p_out,))

                        is_valid, msg = validate_aten_combination(
                            op, (a, b), {}, combo, self.world_size, mesh
                        )

                        # Check if this combo matches any valid rule
                        should_be_valid = (input_plcs, (p_out,)) in valid_rules

                        if should_be_valid:
                            self.assertTrue(
                                is_valid,
                                lambda _m: f"{_m}\n{op}: {p1},{p2}->{p_out} should be valid but got: {msg}",
                            )
                        else:
                            self.assertIs(
                                is_valid,
                                False,
                                lambda msg: f"{msg}\n{op}: {p1},{p2}->{p_out} should be invalid",
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
        add, args, kwargs = torch.ops.aten.add.Tensor, (a, b), {"alpha": -1}

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))

        # R + alpha*P(max) where alpha=-1 should produce P(min)
        # because -max(x) = min(-x)
        combo_valid = ((Replicate(), Partial("max")), (Partial("min"),))
        is_valid, msg = validate_aten_combination(
            add, args, kwargs, combo_valid, self.world_size, mesh
        )
        self.assertTrue(
            is_valid,
            lambda _m: f"{_m}\nR,Pmax->Pmin with alpha=-1 should be valid: {msg}",
        )

        # R + alpha*P(max) where alpha=-1 should NOT produce P(max)
        combo_invalid = ((Replicate(), Partial("max")), (Partial("max"),))
        is_valid, msg = validate_aten_combination(
            add, args, kwargs, combo_invalid, self.world_size, mesh
        )
        self.assertIs(is_valid, False, "R,Pmax->Pmax with alpha=-1 should be invalid")

        # Similarly, P(max) + alpha*R where alpha=-1 should produce P(max)
        # because we're subtracting a replicated value from P(max)
        combo_pmax_minus_r = ((Partial("max"), Replicate()), (Partial("max"),))
        is_valid, msg = validate_aten_combination(
            add, args, kwargs, combo_pmax_minus_r, self.world_size, mesh
        )
        self.assertTrue(
            is_valid,
            lambda _m: f"{_m}\nPmax,R->Pmax with alpha=-1 should be valid: {msg}",
        )


class TestPartialPieces(TestCase):
    """Test the per-rank pieces built for Partial inputs (_partial_pieces)."""

    def test_partial_sum_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        pieces = _partial_pieces(tensor, Partial("sum"), 2)
        self.assertTrue(torch.allclose(pieces[0] + pieces[1], tensor))

    def test_partial_avg_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        pieces = _partial_pieces(tensor, Partial("avg"), 2)
        self.assertTrue(torch.allclose((pieces[0] + pieces[1]) / 2, tensor))

    def test_partial_min_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        pieces = _partial_pieces(tensor, Partial("min"), 2)
        self.assertEqual(torch.minimum(pieces[0], pieces[1]), tensor, atol=0, rtol=0)

    def test_partial_max_reduces_correctly(self):
        tensor = torch.randn(4, 3)
        pieces = _partial_pieces(tensor, Partial("max"), 2)
        self.assertEqual(torch.maximum(pieces[0], pieces[1]), tensor, atol=0, rtol=0)

    def test_tensor_idx_produces_different_sum_ratios(self):
        """
        Verify that different tensor_idx values produce different P(sum) ratios.

        This is critical for catching invalid rules like mul P(sum),P(sum)->P(sum).
        If both tensors used the same 0.5/0.5 split, some invalid combinations
        might accidentally produce correct results.
        """
        tensor = torch.ones(4, 3)
        pieces0 = _partial_pieces(tensor, Partial("sum"), 2, tensor_idx=0)
        pieces1 = _partial_pieces(tensor, Partial("sum"), 2, tensor_idx=1)

        # Both should reduce to the same original tensor
        self.assertTrue(torch.allclose(pieces0[0] + pieces0[1], tensor))
        self.assertTrue(torch.allclose(pieces1[0] + pieces1[1], tensor))

        # But the per-rank values should be DIFFERENT
        # (tensor_idx=0 uses ratio 0.6, tensor_idx=1 uses ratio 0.7)
        self.assertFalse(
            torch.allclose(pieces0[0], pieces1[0]),
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
        pieces0 = _partial_pieces(tensor, Partial("max"), 2, tensor_idx=0)
        pieces1 = _partial_pieces(tensor, Partial("max"), 2, tensor_idx=1)

        # Both should reduce to the same original tensor
        self.assertEqual(torch.maximum(*pieces0), tensor, atol=0, rtol=0)
        self.assertEqual(torch.maximum(*pieces1), tensor, atol=0, rtol=0)

        # The offset patterns should be different (shifted checkerboard)
        self.assertFalse(
            torch.allclose(pieces0[0], pieces1[0]),
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
                            lambda msg: f"{msg}\nMask should alternate along dim 0 at [{':,'}{j},{k},{l}], "
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
            r0, r1 = _partial_pieces(tensor, Partial(reduce_op), 2)
            # The actual offset applied should exceed value_range
            max_offset = (r0 - r1).abs().max().item()
            self.assertGreater(
                max_offset,
                value_range,
                lambda msg: f"{msg}\nP({reduce_op}) offset {max_offset:.1f} should exceed "
                f"value range {value_range:.1f}",
            )
            # Ranks should disagree on argmin/argmax
            self.assertNotEqual(
                r0.argmin().item(),
                r1.argmin().item(),
                lambda msg: f"{msg}\nP({reduce_op}) ranks should disagree on argmin with adaptive offset",
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
        _partial_pieces must create local values with mixed signs to expose
        this, not just proportional splits (tensor * ratio) that preserve the
        sign pattern.
        """
        t = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        combo = ((Partial("sum"),), (Partial("sum"),))

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
        is_valid, msg = validate_aten_combination(
            torch.ops.aten.abs.default, (t,), {}, combo, self.world_size, mesh
        )

        self.assertIs(
            is_valid,
            False,
            "abs P(sum)->P(sum) should be invalid (abs is non-linear)",
        )

    def test_abs_pavg_pavg_is_invalid(self):
        """P(avg)->P(avg) is NOT valid for abs, same reasoning as P(sum)."""
        t = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        combo = ((Partial("avg"),), (Partial("avg"),))

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
        is_valid, msg = validate_aten_combination(
            torch.ops.aten.abs.default, (t,), {}, combo, self.world_size, mesh
        )

        self.assertIs(
            is_valid,
            False,
            "abs P(avg)->P(avg) should be invalid (abs is non-linear)",
        )

    def test_all_zero_output_false_positive(self):
        """
        All-zero ground truth makes every placement trivially validate.

        Zeros are a fixed point of all reduce operations (sum, max, min),
        so validate_aten_combination cannot distinguish valid from invalid
        rules. This is a known limitation: the missing-rule search skips
        all-zero samples to avoid hundreds of false positive rules.

        We use zeros_like as the test op because it always produces zeros
        regardless of input values, unlike mul(zeros, x) which the offsets in
        _partial_pieces handle.
        """
        t = torch.randn(8, 4)

        # P(sum)->P(sum) is NOT valid for zeros_like, but with all-zero
        # output it trivially passes: sum(0,0)=0=ground_truth.
        combo = ((Partial("sum"),), (Partial("sum"),))

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
        is_valid, msg = validate_aten_combination(
            torch.ops.aten.zeros_like.default, (t,), {}, combo, self.world_size, mesh
        )

        # This demonstrates the false positive: validate_aten_combination says
        # valid, even though the rule is invalid for non-zero inputs.
        self.assertTrue(
            is_valid,
            "Expected True (false positive) for all-zero output, showing "
            "why the missing-rule search must skip such samples",
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
        argmin = torch.ops.aten.argmin.default

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
        for reduce_op in ("min", "max"):
            combo = ((Partial(reduce_op),), (Replicate(),))
            is_valid, msg = validate_aten_combination(
                argmin, (t, 0, True), {}, combo, self.world_size, mesh
            )
            self.assertIs(
                is_valid,
                False,
                lambda _m: f"{_m}\nargmin P({reduce_op})->R should be invalid "
                f"(index op, ranks disagree): {msg}",
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
        Query claims for a unary op (softplus) through the decomposition path,
        whose decomposition uses only registered ops. softplus's own strategy
        is removed temporarily to force that path.
        """
        try:
            from torch.distributed.tensor._decompositions import DecompShardingStrategy
        except ImportError:
            self.skipTest("_decompositions module not available")

        aten_softplus = torch.ops.aten.softplus.default
        propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        self.assertTrue(DecompShardingStrategy.has_decomp(aten_softplus))
        with (
            unittest.mock.patch.dict(propagator.op_single_dim_strategy_funcs),
            unittest.mock.patch.dict(propagator.op_strategy_funcs),
        ):
            propagator.op_single_dim_strategy_funcs.pop(aten_softplus, None)
            propagator.op_strategy_funcs.pop(aten_softplus, None)
            t = torch.randn(8, 4)
            with LocalTensorMode(frozenset(range(self.world_size))):
                mesh = init_device_mesh("cpu", (self.world_size,))
            for dim in (0, 1):
                claims = query_runtime_claims(
                    aten_softplus, (t,), {}, (Shard(dim),), mesh
                )
                self.assertIn((Shard(dim),), [c.output_placements for c in claims])

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


class TestQueryRuntimeClaimsKwargs(TestCase):
    """Test that query_runtime_claims forwards kwargs to strategy functions."""

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

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
            a, b = torch.randn(4, 3), torch.randn(4, 3)
            with LocalTensorMode(frozenset(range(self.world_size))):
                mesh = init_device_mesh("cpu", (self.world_size,))
            claims = query_runtime_claims(
                aten_add, (a, b), {"alpha": -1}, (Replicate(), Partial("max")), mesh
            )
            self.assertEqual(
                [c.output_placements for c in claims],
                [(Partial("min"),)],
                "With alpha=-1, the strategy should produce P(min) output "
                "but got something else — kwargs were not forwarded",
            )
        finally:
            if original is not None:
                propagator.op_single_dim_strategy_funcs[aten_add] = original
            else:
                propagator.op_single_dim_strategy_funcs.pop(aten_add, None)


class TestCompareRules(TestCase):
    """Test _compare_rules populates per-op statistics correctly."""

    def test_true_positives_tracked_by_op(self):
        from torch.distributed.tensor._ops.strategy_validation import (
            _compare_rules,
            ComparisonStats,
        )

        combo_a = (("S(0)",), ("R",))
        combo_b = (("S(1)",), ("R",))
        stats = ComparisonStats()
        _compare_rules(
            ground_truth_valid={combo_a, combo_b},
            dtensor_rules={combo_a, combo_b},
            input_shapes=((4, 8),),
            output_shapes=((4, 8),),
            sample_idx=0,
            scalar_args=(),
            scalar_kwargs={},
            aten_op=torch.ops.aten.relu.default,
            variant="",
            stats=stats,
        )
        self.assertEqual(stats.true_positives, 2)
        self.assertEqual(stats.true_positives_by_op, {"aten.relu.default": 2})
        self.assertEqual(len(stats.false_positives), 0)
        self.assertEqual(len(stats.false_negatives), 0)

    def test_per_op_breakdown_multiple_aten_ops(self):
        """Simulate an opinfo that dispatches to different aten variants per sample."""
        from torch.distributed.tensor._ops.strategy_validation import (
            _compare_rules,
            ComparisonStats,
        )

        stats = ComparisonStats()

        # Sample 0: aten.max.dim returns (values, indices)
        combo_2out = (("S(0)",), ("R", "R"))
        _compare_rules(
            ground_truth_valid={combo_2out},
            dtensor_rules={combo_2out},
            input_shapes=((4, 8),),
            output_shapes=((4, 8), (4, 8)),
            sample_idx=0,
            scalar_args=(1,),
            scalar_kwargs={},
            aten_op=torch.ops.aten.max.dim,
            variant="",
            stats=stats,
        )
        # Sample 1: aten.native_layer_norm returns (output, mean, rstd)
        combo_3out = (("S(0)",), ("R", "R", "R"))
        _compare_rules(
            ground_truth_valid={combo_3out},
            dtensor_rules={combo_3out},
            input_shapes=((4, 8),),
            output_shapes=((4, 8), (4,), (4,)),
            sample_idx=1,
            scalar_args=(),
            scalar_kwargs={},
            aten_op=torch.ops.aten.native_layer_norm.default,
            variant="",
            stats=stats,
        )

        self.assertEqual(stats.true_positives, 2)
        self.assertEqual(
            stats.true_positives_by_op,
            {"aten.max.dim": 1, "aten.native_layer_norm.default": 1},
        )

    def test_false_positives_and_negatives_not_in_by_op(self):
        """true_positives_by_op only tracks true positives, not FP/FN."""
        from torch.distributed.tensor._ops.strategy_validation import (
            _compare_rules,
            ComparisonStats,
        )

        gt_only = (("S(0)",), ("R",))
        dt_only = (("S(1)",), ("R",))
        stats = ComparisonStats()
        _compare_rules(
            ground_truth_valid={gt_only},
            dtensor_rules={dt_only},
            input_shapes=((4, 8),),
            output_shapes=((4, 8),),
            sample_idx=0,
            scalar_args=(),
            scalar_kwargs={},
            aten_op=torch.ops.aten.relu.default,
            variant="",
            stats=stats,
        )
        self.assertEqual(stats.true_positives, 0)
        self.assertEqual(stats.true_positives_by_op, {})
        self.assertEqual(len(stats.false_positives), 1)
        self.assertEqual(len(stats.false_negatives), 1)


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
                lambda msg: f"{msg}\nsplit should have runnable samples, got skip_reasons={stats.skip_reasons}",
            )
            self.assertEqual(len(stats.false_positives), 0)

        self._with_even_sizes(run)

    def test_compare_operator_runtime_schema_ops(self):
        """Ops with RuntimeSchemaInfo (non-tensor positional args) should find DTensor rules."""
        from torch.distributed.tensor._ops.strategy_validation import compare_operator

        runtime_schema_ops = ["flip", "roll"]
        for op_name in runtime_schema_ops:
            with self.subTest(op=op_name):

                def run(name=op_name):
                    stats = compare_operator(
                        name,
                        device="cpu",
                        dtype=torch.float32,
                        world_size=self.world_size,
                        incorrect_only=True,
                    )
                    self.assertGreater(
                        stats.true_positives,
                        0,
                        lambda msg: f"{msg}\n{name} should have DTensor rules (true_positives > 0), "
                        f"got skip_reasons={stats.skip_reasons}",
                    )

                self._with_even_sizes(run)


class TestAtenLevelValidation(TestCase):
    """Tests for aten-level validation (validate_aten_combination + allow_composite mode)."""

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

    def test_validate_aten_combination_shard_add(self):
        """S(0), S(0) -> S(0) valid; S(0), S(0) -> R invalid for aten.add."""
        a = torch.randn(8)
        b = torch.randn(8)

        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))

        valid, msg = validate_aten_combination(
            torch.ops.aten.add.Tensor,
            (a, b),
            {},
            ((Shard(0), Shard(0)), (Shard(0),)),
            self.world_size,
            mesh,
        )
        self.assertTrue(valid, msg)

        valid, _ = validate_aten_combination(
            torch.ops.aten.add.Tensor,
            (a, b),
            {},
            ((Shard(0), Shard(0)), (Replicate(),)),
            self.world_size,
            mesh,
        )
        self.assertIs(valid, False)

    def test_allow_composite_eliminates_non_1to1_skips(self):
        """allow_composite mode should validate decomposed ops that default mode skips."""
        from torch.distributed.tensor._ops.strategy_validation import compare_operator

        def run():
            default = compare_operator(
                "inner",
                device="cpu",
                dtype=torch.float32,
                world_size=self.world_size,
                max_samples=3,
                incorrect_only=True,
            )
            allow_composite = compare_operator(
                "inner",
                device="cpu",
                dtype=torch.float32,
                world_size=self.world_size,
                max_samples=3,
                incorrect_only=True,
                allow_composite=True,
            )
            self.assertGreater(default.skip_reasons.get("non-1:1 aten mapping", 0), 0)
            self.assertEqual(
                allow_composite.skip_reasons.get("non-1:1 aten mapping", 0), 0
            )
            self.assertGreaterEqual(
                allow_composite.total_samples, default.total_samples
            )

        self._with_even_sizes(run)


class TestSpecialValuePartialInput(TestCase):
    """Partial pieces reduce to inputs with special values and integer extremes.

    Integer and P(min)/P(max) pieces reduce exactly; floating P(sum) pieces
    reduce up to rounding (the validator uses the reduced value as the input).
    """

    def test_pieces_reduce_to_input(self):
        inputs = [
            torch.tensor([1.0, math.nan, math.inf, -math.inf, -0.0, 2.5, -3.0, 0.0]),
            torch.randn(3, 5),
            torch.tensor([127, -128, 0, 5, -7, 100], dtype=torch.int8),
            torch.tensor([0, 255, 1, 128, 7], dtype=torch.uint8),
            torch.tensor([2**63 - 1, -(2**63), 0, 12345], dtype=torch.int64),
        ]
        for t, reduce_op, world_size, adversarial in itertools.product(
            inputs, ("sum", "min", "max"), (2, 3, 4), (False, True)
        ):
            with self.subTest(
                dtype=t.dtype, op=reduce_op, ws=world_size, adversarial=adversarial
            ):
                pieces = _partial_pieces(
                    t, Partial(reduce_op), world_size, adversarial=adversarial
                )
                self.assertTrue(all(p.dtype == t.dtype for p in pieces))
                exact = reduce_op != "sum" or not t.is_floating_point()
                tol = {"atol": 0, "rtol": 0} if exact else {}
                self.assertEqual(_local_reduce(reduce_op, pieces), t, **tol)

    def test_adversarial_pieces(self):
        """Losing ranks hold -inf for P(max); NaN under P(sum) becomes +inf/-inf."""
        t = torch.tensor([1.0, 2.0, math.nan, -1.0])
        pieces = torch.stack(_partial_pieces(t, Partial("max"), 2, adversarial=True))
        self.assertEqual(pieces.isneginf().sum(0)[[0, 1, 3]].tolist(), [1, 1, 1])
        pieces = _partial_pieces(t, Partial("sum"), 2, adversarial=True)
        self.assertEqual(sorted(p[2].item() for p in pieces), [-math.inf, math.inf])

    def test_wraparound_pieces_leave_the_dtype_range(self):
        inputs = [
            torch.tensor([1, 2, -3, 4], dtype=torch.int8),
            torch.tensor([-100, 0, 5, 127], dtype=torch.int8),
            torch.tensor([0, 3, 200, 254], dtype=torch.uint8),
            torch.tensor([7, -(2**63), 2**63 - 1, 0], dtype=torch.int64),
        ]
        for t, world_size in itertools.product(inputs, (2, 3, 4)):
            with self.subTest(dtype=t.dtype, ws=world_size):
                pieces = _partial_pieces(t, Partial("sum"), world_size, wraparound=True)
                self.assertEqual(_local_reduce("sum", pieces), t)
                true_sums = [sum(int(p[i]) for p in pieces) for i in range(t.numel())]
                self.assertNotEqual(true_sums, t.tolist())
        # -1 is the one int8 value two in-range pieces cannot wrap to.
        minus_one = torch.tensor([-1], dtype=torch.int8)
        pieces = _partial_pieces(minus_one, Partial("sum"), 2, wraparound=True)
        self.assertIsNone(pieces)

    def test_unconstructible_partials(self):
        self.assertIsNone(
            _partial_pieces(torch.ones(4, dtype=torch.int32), Partial("avg"), 2)
        )
        self.assertIsNone(_partial_pieces(torch.ones(4), Partial("product"), 2))
        self.assertIsNone(
            _partial_pieces(torch.ones(4, dtype=torch.bool), Partial("max"), 2)
        )


class TestInputVariants(TestCase):
    """Value variants respect operand roles and keep IEEE-policy cases out."""

    def test_specials_skip_other_operands_of_linear_partials(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        tensors = [("a", a), ("b", b)]
        variant = _Variant("special-nan", specials="nan")
        inputs, _ = _materialize_inputs(
            tensors, (Partial("sum"), Replicate()), 2, variant, [False, False]
        )
        self.assertTrue(inputs[0].isnan().any())
        self.assertEqual(inputs[1], b)
        inputs, _ = _materialize_inputs(
            tensors, (Partial("max"), Replicate()), 2, variant, [False, False]
        )
        self.assertTrue(inputs[1].isnan().any())

    def test_adversarial_keeps_other_operands_finite(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        variant = _Variant("adversarial", specials="nan", adversarial=True)
        inputs, pieces = _materialize_inputs(
            [("a", a), ("b", b)],
            (Partial("max"), Replicate()),
            2,
            variant,
            [False, False],
        )
        self.assertEqual(inputs[1], b)
        self.assertTrue(torch.stack(pieces[0]).isneginf().any())

    def test_constrained_operands_are_not_modified(self):
        values = torch.tensor([0.5, 1.5, 2.5, 3.5])
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0])
        constrained = _constrained_operands(
            torch.ops.aten.bucketize.Tensor, (values, boundaries), {}
        )
        self.assertEqual(constrained, [False, True])
        for variant in (
            _Variant("finite", negate=True),
            _Variant("special-nan", specials="nan"),
        ):
            inputs, _ = _materialize_inputs(
                [("v", values), ("b", boundaries)],
                (Replicate(), Shard(0)),
                2,
                variant,
                constrained,
            )
            self.assertEqual(inputs[1], boundaries)
        self.assertEqual(
            _constrained_operands(
                torch.ops.aten.index_select.default,
                (torch.randn(4, 3), 0, torch.tensor([0, 2])),
                {},
            ),
            [False, True],
        )
        ints = torch.ones(4, 4, dtype=torch.int64)
        self.assertEqual(
            _constrained_operands(torch.ops.aten.mm.default, (ints, ints), {}),
            [False, False],
        )

    def test_random_specials_follow_input_device(self):
        t = torch.ones(8, device="meta")
        out = _inject_specials(t, "random", torch.Generator().manual_seed(0))
        self.assertEqual(out.device, t.device)

    def test_scalar_variants(self):
        """Directed values cross (value > threshold) and Scalar args also take ints."""
        x = torch.randn(4)
        variants = _scalar_variants(
            torch.ops.aten.threshold.default,
            (x, 1.0, 0.0),
            {},
            ValidationOptions(scalar_variants=6),
        )
        labels = [label for label, _, _ in variants]
        self.assertEqual(len(variants), 6)
        self.assertIn("scalar:threshold=0.0,value=2.0", labels)
        self.assertTrue(any(isinstance(args[1], int) for _, args, _ in variants))
        # Integer tensors take integer directed values (float alpha is rejected).
        ints = torch.arange(4)
        alphas = [
            kwargs["alpha"]
            for _, _, kwargs in _scalar_variants(
                torch.ops.aten.add.Tensor,
                (ints, ints),
                {},
                ValidationOptions(scalar_variants=6),
            )
        ]
        self.assertEqual(sorted(alphas), [-2, -1, 0, 1, 2, 3])
        # Omitted optional float args are passed positionally, as the
        # dispatcher would pass them.
        variants = _scalar_variants(
            torch.ops.aten.nan_to_num.default,
            (x,),
            {},
            ValidationOptions(scalar_variants=2),
        )
        _, args, kwargs = variants[0]
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 4)
        self.assertTrue(all(isinstance(a, float) for a in args[1:]))
        # Scale factors and numerical constants are not fuzzed.
        self.assertEqual(
            _scalar_variants(
                torch.ops.aten.native_batch_norm.default,
                (x, None, None, None, None, True, 0.1, 1e-5),
                {},
                ValidationOptions(scalar_variants=2),
            ),
            [],
        )


class TestRuntimeClaims(TestCase):
    """query_runtime_claims and uneven-shard validation follow the runtime."""

    world_size = 4

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _mesh(self, world_size):
        with LocalTensorMode(frozenset(range(world_size))):
            return init_device_mesh("cpu", (world_size,))

    def test_mean_partial_avg_only_on_even_shards(self):
        """The runtime filter rejects mean's S(d) -> P(avg) when shards are uneven."""
        mesh = self._mesh(2)

        def outputs(t):
            claims = query_runtime_claims(
                torch.ops.aten.mean.dim, (t, [0]), {}, (Shard(0),), mesh
            )
            return [c.output_placements for c in claims]

        self.assertIn((Partial("avg"),), outputs(torch.randn(8, 4)))
        self.assertNotIn((Partial("avg"),), outputs(torch.randn(7, 4)))

    def test_view_claim_rewrites_shape_args(self):
        claims = query_runtime_claims(
            torch.ops.aten.view.default,
            (torch.randn(8, 4), [32]),
            {},
            (Shard(0),),
            self._mesh(2),
        )
        shard_claims = [c for c in claims if c.output_placements == (Shard(0),)]
        self.assertEqual(len(shard_claims), 1)
        self.assertEqual(list(shard_claims[0].local_args[1]), [16])

    def test_uneven_and_empty_shards(self):
        aten_add = torch.ops.aten.add.Tensor
        for size, world_size in ((7, 3), (2, 3), (5, 4)):
            a, b = torch.randn(size, 3), torch.randn(size, 3)
            mesh = self._mesh(world_size)
            valid, msg = validate_aten_combination(
                aten_add,
                (a, b),
                {},
                ((Shard(0), Shard(0)), (Shard(0),)),
                world_size,
                mesh,
            )
            self.assertTrue(valid, msg)
            valid, _ = validate_aten_combination(
                aten_add,
                (a, b),
                {},
                ((Shard(0), Shard(0)), (Partial("sum"),)),
                world_size,
                mesh,
            )
            self.assertIs(valid, False)


class TestSizeProfiles(TestCase):
    def test_profile_reaches_opinfo_definitions(self):
        """OpInfo definition modules import S/M/L/XS by value; all of them are patched and restored."""
        from torch.testing._internal.opinfo.definitions import linalg

        before = linalg.S
        with _opinfo_sizes("small"):
            self.assertEqual(linalg.S, 3)
        self.assertEqual(linalg.S, before)


class TestComparator(TestCase):
    """Non-finite values and dtypes must match exactly."""

    def test_nonfinite_and_dtype_mismatches(self):
        inf, one = torch.tensor([math.inf]), torch.tensor([1.0])
        self.assertTrue(_mismatch(inf, one))
        self.assertTrue(_mismatch(inf, -inf))
        nan = torch.tensor([math.nan])
        self.assertEqual(_mismatch(nan, nan.clone()), "")
        self.assertTrue(_mismatch(torch.tensor([16777216.0]), torch.tensor([16777217])))

    def test_complex_values(self):
        """Closeness by magnitude as torch.isclose; per component where a component is infinite."""
        a = torch.tensor([complex(math.inf, 0.5), complex(1.0, -2.0)])
        self.assertEqual(_mismatch(a.conj(), a.conj()), "")
        b = torch.tensor([complex(math.inf, 0.5 + 1e-7), complex(1.0, -2.0)])
        self.assertEqual(_mismatch(a, b), "")
        self.assertTrue(_mismatch(a, a.conj().resolve_conj()))
        small_imag = torch.tensor([complex(1e3, 1e-3)])
        close_imag = torch.tensor([complex(1e3, 1.5e-3)])
        self.assertEqual(_mismatch(small_imag, close_imag), "")
        nan = torch.tensor([complex(math.nan, 1.0)])
        self.assertEqual(_mismatch(nan, torch.tensor([complex(math.nan, 2.0)])), "")


@torch.library.custom_op("sv_test::sign_like", mutates_args=())
def _sign_like(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)


@_sign_like.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::exp_like", mutates_args=())
def _exp_like(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


@_exp_like.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::threshold_like", mutates_args=())
def _threshold_like(x: torch.Tensor, threshold: float, value: float) -> torch.Tensor:
    return torch.threshold(x, threshold, value)


@_threshold_like.register_fake
def _(x, threshold, value):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::affine", mutates_args=())
def _affine(x: torch.Tensor) -> torch.Tensor:
    return x * 1e7 + 1


@_affine.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::affine_abs", mutates_args=())
def _affine_abs(x: torch.Tensor) -> torch.Tensor:
    return x * 1e7 + x.abs()


@_affine_abs.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::floor_like", mutates_args=())
def _floor_like(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x)


@_floor_like.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::step_diff", mutates_args=())
def _step_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 256 * (x - y) + (x < y).to(x.dtype)


@_step_diff.register_fake
def _(x, y):
    return torch.empty_like(x)


@torch.library.custom_op("sv_test::copy_like", mutates_args=())
def _copy_like(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_copy_like.register_fake
def _(x):
    return torch.empty_like(x)


_sv_test_lib = torch.library.Library("sv_test", "FRAGMENT")
_sv_test_lib.define("neg_max(Tensor x, Tensor y) -> Tensor")
_sv_test_lib.impl(
    "neg_max", lambda x, y: torch.maximum(x, -y), "CompositeImplicitAutograd"
)


class TestDetectsIncorrectRules(TestCase):
    """
    Custom ops get a P(max) -> P(max) rule: wrong for sign (sign(NaN) = 0) and
    for threshold with value > threshold, right for exp. affine (x * 1e7 + 1)
    and affine_abs (x * 1e7 + |x|) get a P(sum) -> P(sum) rule, wrong by
    amounts much smaller than the pieces. The validator must flag the wrong
    rules only under the input variant that exposes them. floor_like and
    step_diff get wrong P(sum) rules whose float32 failures vanish in 64-bit:
    those must stay unresolved, never correct.
    """

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)
        propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        self.ops = [
            torch.ops.sv_test.sign_like.default,
            torch.ops.sv_test.exp_like.default,
            torch.ops.sv_test.threshold_like.default,
        ]
        rules = [(op, Partial("max")) for op in self.ops]
        rules.append((torch.ops.sv_test.affine.default, Partial("sum")))
        rules.append((torch.ops.sv_test.affine_abs.default, Partial("sum")))
        rules.append((torch.ops.sv_test.floor_like.default, Partial("sum")))
        registry = unittest.mock.patch.dict(propagator.op_single_dim_strategy_funcs)
        registry.start()
        self.addCleanup(registry.stop)
        for op, placement in rules:
            register_single_dim_strategy(op)(
                lambda op, args, kwargs, p=placement: [[p, p]]
            )
        register_single_dim_strategy(torch.ops.sv_test.step_diff.default)(
            lambda op, args, kwargs: [[Partial("sum")] * 3]
        )

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _audit(self, op, args, options, kwargs=None):
        stats = ComparisonStats()
        kwargs = kwargs or {}
        _validate_captured_call(
            op,
            args,
            kwargs,
            op(*args, **kwargs),
            self.world_size,
            incorrect_only=True,
            verbose=False,
            sample_idx=0,
            variant="",
            stats=stats,
            sample=None,
            options=options,
        )
        return stats

    def _incorrect(self, op, args, options):
        return [d.trigger for d in self._audit(op, args, options).false_positives]

    def test_nan_dependent_rule(self):
        x = torch.randn(4, 6)
        sign_like, exp_like, _ = self.ops
        self.assertEqual(self._incorrect(sign_like, (x,), ValidationOptions()), [])
        special = ValidationOptions(special_values=True)
        self.assertEqual(self._incorrect(sign_like, (x,), special), ["nan-partial"])
        stats = self._audit(exp_like, (x,), special)
        self.assertEqual(stats.false_positives, [])
        self.assertGreater(stats.true_positives, 0)
        # A NaN already in the captured input is classified by the pieces.
        x[0, 0] = math.nan
        self.assertEqual(
            self._incorrect(sign_like, (x,), ValidationOptions()), ["nan-partial"]
        )

    def test_scalar_dependent_rule(self):
        threshold_like = self.ops[2]
        args = (torch.linspace(-3, 3, 24).reshape(4, 6), 1.0, 0.0)
        self.assertEqual(self._incorrect(threshold_like, args, ValidationOptions()), [])
        triggers = self._incorrect(
            threshold_like, args, ValidationOptions(scalar_variants=6)
        )
        self.assertIn("scalar:threshold=0.0,value=2.0 finite", triggers)

    def test_rounding_tolerance_does_not_hide_small_errors(self):
        """The errors are far below the pieces' magnitude, so only the 64-bit recheck of the same pieces can tell them from rounding."""
        ops = (torch.ops.sv_test.affine.default, torch.ops.sv_test.affine_abs.default)
        for op, dtype in itertools.product(ops, (torch.float32, torch.float64)):
            x = torch.zeros(4, 6, dtype=dtype)
            self.assertEqual(self._incorrect(op, (x,), ValidationOptions()), ["finite"])

    def test_mismatch_that_vanishes_in_64bit_is_unresolved(self):
        """A float32 mismatch that disappears in 64-bit may be a branch flip, so the rule is not counted correct.

        floor at -19 and a step at x == y: the exact 64-bit sum of the float32
        pieces moves the reference across the branch.
        """
        floor_like = torch.ops.sv_test.floor_like.default
        step_diff = torch.ops.sv_test.step_diff.default
        x = torch.tensor([-19.0])
        for op, args in ((floor_like, (x,)), (step_diff, (x, x.clone()))):
            with self.subTest(op=str(op)):
                stats = self._audit(op, args, ValidationOptions())
                self.assertEqual(stats.true_positives, 0)
                self.assertEqual(stats.false_positives, [])
                reason = "unresolved: mismatch vanishes in 64-bit"
                self.assertEqual(stats.untestable_claims, {reason: 1})

    def test_per_rank_overflow_is_unresolved(self):
        """P(sum) pieces that overflow when scaled are neither accepted nor reported as incorrect, even when other variants pass."""
        mul = torch.ops.aten.mul.Scalar
        cases = (
            (torch.tensor([2.0]), 1e38),
            (torch.tensor([0.25], dtype=torch.float16), 1e5),
            (torch.tensor([2.0], dtype=torch.bfloat16), 1e38),
        )
        for (x, scalar), special in itertools.product(cases, (False, True)):
            with self.subTest(dtype=x.dtype, special_values=special):
                options = ValidationOptions(special_values=special)
                stats = self._audit(mul, (x, scalar), options)
                self.assertEqual(stats.false_positives, [])
                reason = "unresolved: mismatch vanishes in 64-bit"
                self.assertIn(reason, stats.untestable_claims)

    def test_64bit_recheck_widens_dtype_arguments(self):
        """cumsum(dtype=float16) is re-run with dtype=float64, so float16 rounding of its correct linear rules is unresolved, not incorrect."""
        cumsum = torch.ops.aten.cumsum.default
        x = torch.randn(4, 4, 4, dtype=torch.float16)
        kwargs = {"dtype": torch.float16}
        stats = self._audit(cumsum, (x, 1), ValidationOptions(), kwargs)
        self.assertEqual(stats.false_positives, [])
        reason = "unresolved: mismatch vanishes in 64-bit"
        self.assertEqual(stats.untestable_claims, {reason: 1})


class TestClaimDiscovery(TestCase):
    """Claims of every kind of strategy are found, and failures to query them are reported."""

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=4)
        with LocalTensorMode(frozenset(range(2))):
            self.mesh = init_device_mesh("cpu", (2,))

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _audit(self, op, args, world_size=2, incorrect_only=True, options=None):
        stats = ComparisonStats()
        _validate_aten_op_for_sample(
            op,
            args,
            {},
            op(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args]),
            world_size,
            incorrect_only=incorrect_only,
            verbose=False,
            sample_idx=0,
            variant="",
            stats=stats,
            options=options,
        )
        return stats

    def test_uneven_joint_claims_are_discovered(self):
        """Claims for jointly sharded inputs whose candidates depend on the placements (empty shard on rank 2)."""
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        options = [
            get_1d_input_placements_for_tensor(t, include_partial=True) for t in (x, y)
        ]
        with LocalTensorMode(frozenset(range(3))):
            mesh = init_device_mesh("cpu", (3,))
            combos, complete = claimed_input_combinations(
                torch.ops.aten.add.Tensor, (x, y), {}, options, mesh
            )
        self.assertTrue(complete)
        self.assertIn((Shard(0), Shard(0)), combos)

    def test_missing_rule_search_on_uneven_and_size_one_dims(self):
        """The full search runs on uneven shards and skips outputs sharded on size-1 dims."""
        add = torch.ops.aten.add.Tensor
        stats = self._audit(add, (torch.randn(7), torch.randn(7)), 3, False)
        self.assertEqual(stats.false_positives, [])
        self.assertGreater(stats.true_positives, 0)
        unsqueeze = torch.ops.aten.unsqueeze.default
        stats = self._audit(unsqueeze, (torch.randn(4, 3), 0), 2, False)
        self.assertEqual(stats.false_positives, [])
        self.assertNotIn(
            "S(0)", {d.output_placements[0] for d in stats.false_negatives}
        )

    def test_in_place_ops_get_independent_pieces(self):
        """Wraparound pieces must not alias across ranks: an in-place op would update them once per rank."""
        special = ValidationOptions(special_values=True)
        x = torch.arange(-3, 3)
        stats = self._audit(torch.ops.aten.mul_.Scalar, (x, 3), 3, True, special)
        self.assertEqual(stats.false_positives, [])
        self.assertGreater(stats.true_positives, 0)

    def test_decomposition_mixed_placement_claim(self):
        """maximum(x, -y) keeps P(max) for (P(max), P(min)) inputs, which no single-input seed asks for."""
        op = torch.ops.sv_test.neg_max.default
        x, y = torch.randn(4, 6), torch.randn(4, 6)
        options = [
            get_1d_input_placements_for_tensor(t, include_partial=True) for t in (x, y)
        ]
        with LocalTensorMode(frozenset(range(2))):
            combos, complete = claimed_input_combinations(
                op, (x, y), {}, options, self.mesh
            )
        self.assertTrue(complete)
        self.assertIn((Partial("max"), Partial("min")), combos)
        stats = self._audit(op, (x, y))
        self.assertEqual(stats.false_positives, [])
        self.assertGreater(stats.true_positives, 0)

    def test_dtype_mismatch_is_not_a_value_mismatch(self):
        """A wrong output dtype is reported as such (never re-run in 64-bit as a possible rounding error)."""
        pieces = {r: torch.zeros(2, dtype=torch.float16) for r in range(2)}
        with LocalTensorMode(frozenset(range(2))):
            # pyrefly: ignore [bad-argument-type]
            out = LocalTensor(pieces)
            want = torch.zeros(4)
            valid, msg = _compare_outputs(out, want, (Shard(0),), self.mesh, 2)
        self.assertIs(valid, False)
        self.assertTrue(msg.startswith("Dtype mismatch"), msg)

    def test_lazy_conjugate_outputs(self):
        """Conjugate views are compared like resolved tensors (aten._conj S(0) -> S(0) is correct)."""
        x = torch.randn(4, 6, dtype=torch.complex64)
        stats = self._audit(torch.ops.aten._conj.default, (x,))
        self.assertEqual(stats.false_positives, [])
        self.assertGreater(stats.true_positives, 0)

    def test_query_failures_are_reported(self):
        a = [torch.randn(4, 6) for _ in range(2)]
        b = [torch.randn(4, 6) for _ in range(2)]
        stats = self._audit(torch.ops.aten._foreach_add.List, (a, b))
        self.assertIn("query: TupleStrategy is not supported", stats.untestable_claims)

    def test_selected_rewrite_failure_is_incorrect(self):
        """Only the candidate dispatch selects is reported; the alternative is untestable."""
        propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        op = torch.ops.sv_test.copy_like.default
        x = torch.randn(8, 4)
        with (
            unittest.mock.patch.dict(propagator.op_single_dim_strategy_funcs),
            unittest.mock.patch.object(
                propagator, "_rewrite_args_for_local", side_effect=IndexError("boom")
            ),
        ):
            register_single_dim_strategy(op)(
                lambda op, args, kwargs: [
                    [Shard(0), Shard(0)],
                    [Partial("sum"), Shard(0)],
                ]
            )
            claims = query_runtime_claims(op, (x,), {}, (Shard(0),), self.mesh)
            stats = self._audit(op, (x,))
        self.assertEqual(sorted(c.selected for c in claims), [False, True])
        failures = [(d.input_placements, d.trigger) for d in stats.false_positives]
        self.assertEqual(failures, [(("S(0)",), "rewrite")])
        self.assertEqual(stats.untestable_claims, {"local arg rewrite raised": 1})


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
            lambda msg: f"{msg}\nModule exited with code {result.returncode}.\n"
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
