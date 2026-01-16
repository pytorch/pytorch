# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for partial placement propagation through pointwise ops.

This file tests whether propagating partial placements through pointwise
operations produces the same result as redistributing all inputs to
Replicate first and then computing.

Uses the experimental `propagate_all_partials_pointwise_strategy` which:
1. Propagates ALL partial types (not just sum/avg)
2. Keeps Replicate inputs as Replicate (no redistribution to Partial)
"""

import os
import sys
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch._ops import OpOverload
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Partial,
    Replicate,
)
from torch.distributed.tensor._ops._math_ops import _NormPartial
from torch.distributed.tensor._ops._pointwise_ops import (
    propagate_all_partials_pointwise_strategy,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    with_comms,
)


# Check for --summarize flag
SUMMARIZE_MODE = "--summarize" in sys.argv
if SUMMARIZE_MODE:
    sys.argv.remove("--summarize")


# All partial types to test: (class, parameter, display_name)
# Includes regular Partial types and NormPartial variants
PARTIAL_TYPES = [
    # Regular Partial types
    (Partial, "sum", "P(sum)"),
    (Partial, "avg", "P(avg)"),
    (Partial, "min", "P(min)"),
    (Partial, "max", "P(max)"),
    (Partial, "product", "P(product)"),
    # NormPartial types (L2 norm ~ sum, inf ~ max, -inf ~ min)
    (_NormPartial, 1, "NP(1)"),
    (_NormPartial, 2, "NP(2)"),
    (_NormPartial, float("inf"), "NP(inf)"),
    (_NormPartial, float("-inf"), "NP(-inf)"),
]

# Unary pointwise ops to test (torch function -> aten op)
UNARY_OPS = [
    (torch.abs, torch.ops.aten.abs.default),
    (torch.neg, torch.ops.aten.neg.default),
    (torch.relu, torch.ops.aten.relu.default),
    (torch.sigmoid, torch.ops.aten.sigmoid.default),
    (torch.tanh, torch.ops.aten.tanh.default),
]

# Binary pointwise ops to test (torch function -> aten op)
BINARY_OPS = [
    (torch.add, torch.ops.aten.add.Tensor),
    (torch.sub, torch.ops.aten.sub.Tensor),
    (torch.mul, torch.ops.aten.mul.Tensor),
    (torch.div, torch.ops.aten.div.Tensor),
    (torch.maximum, torch.ops.aten.maximum.default),
    (torch.minimum, torch.ops.aten.minimum.default),
]

OUTPUT_FILE = "partial_propagation_results.txt"


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_test_data(device_type: str, rank: int) -> dict[str, torch.Tensor]:
    """
    Generate all test data patterns for a given rank.
    Returns a dict with keys: mono1, mono2, varied1, varied2, neg1, neg2
    """
    # Monotonic: lower ranks have smaller values
    mono1 = torch.tensor([[-1.0, -2.0], [-3.0, 4.0]], device=device_type) + rank
    mono2 = torch.tensor([[-5.0, 6.0], [-7.0, -8.0]], device=device_type) + rank

    # Varied: min/max on different ranks for different elements
    if rank % 2 == 0:
        varied1 = torch.tensor([[10.0, 1.0], [1.0, 10.0]], device=device_type)
        varied2 = torch.tensor([[1.0, 10.0], [10.0, 1.0]], device=device_type)
    else:
        varied1 = torch.tensor([[1.0, 10.0], [10.0, 1.0]], device=device_type)
        varied2 = torch.tensor([[10.0, 1.0], [1.0, 10.0]], device=device_type)

    # All negative: all values are negative
    neg1 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device=device_type) - rank
    neg2 = torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device=device_type) - rank

    return {
        "mono1": mono1,
        "mono2": mono2,
        "varied1": varied1,
        "varied2": varied2,
        "neg1": neg1,
        "neg2": neg2,
    }


def generate_replicate_data(device_type: str) -> dict[str, torch.Tensor]:
    """Generate fixed data for Replicate inputs (same across all ranks)."""
    return {
        "pos": torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=device_type),
        "neg": torch.tensor([[-2.0, -3.0], [-4.0, -5.0]], device=device_type),
    }


# =============================================================================
# Strategy Override Context Manager
# =============================================================================


@contextmanager
def override_op_strategies(ops: list[OpOverload]):
    """
    Context manager to temporarily override op strategies with the
    experimental propagate_all_partials_pointwise_strategy.
    """
    propagator = DTensor._op_dispatcher.sharding_propagator

    # Save original strategies
    original_strategies = {}
    for op in ops:
        if op in propagator.op_strategy_funcs:
            original_strategies[op] = propagator.op_strategy_funcs[op]

    # Override with experimental strategy
    for op in ops:
        propagator.op_strategy_funcs[op] = propagate_all_partials_pointwise_strategy

    # Clear the cache so new strategies take effect
    propagator.propagate_op_sharding.cache_clear()

    try:
        yield
    finally:
        # Restore original strategies
        for op in ops:
            if op in original_strategies:
                propagator.op_strategy_funcs[op] = original_strategies[op]
        # Clear cache again
        propagator.propagate_op_sharding.cache_clear()


# =============================================================================
# Result Writing Functions
# =============================================================================

# Data pattern types used in result keys
DATA_PATTERNS = ("mono", "varied", "allneg", "rneg")


def _group_results_by_op(results: dict) -> dict:
    """Group results by op name, collapsing data patterns."""
    by_op = defaultdict(lambda: defaultdict(list))
    for key, (success, error) in results.items():
        op_name = key[0]
        placements = key[1:]
        # Remove the last element if it's a data pattern type
        if placements and placements[-1] in DATA_PATTERNS:
            base_placements = placements[:-1]
            data_type = placements[-1]
        else:
            base_placements = placements
            data_type = None
        by_op[op_name][base_placements].append((data_type, success, error))
    return by_op


def write_results_by_op(f, results, title):
    """
    Write results organized by op name.
    Collapses data patterns - a placement combo passes only if ALL pass.
    Shows which data patterns failed for failed cases.
    """
    by_op = _group_results_by_op(results)

    f.write(f"\n{'=' * 80}\n")
    f.write(f"{title}\n")
    f.write(f"{'=' * 80}\n\n")

    for op_name in sorted(by_op.keys()):
        f.write(f"{op_name}:\n")
        passed, failed = [], []

        for base_placements, results_list in sorted(by_op[op_name].items()):
            placement_str = ", ".join(str(p) for p in base_placements)
            all_passed = all(success for _, success, _ in results_list)

            if all_passed:
                passed.append(f"  [PASS] {placement_str}")
            else:
                failed_patterns = [dt for dt, success, _ in results_list if not success]
                failed_str = (
                    ", ".join(failed_patterns) if failed_patterns else "unknown"
                )
                failed.append(f"  [FAIL] {placement_str} (failed: {failed_str})")

        for line in passed + failed:
            f.write(line + "\n")
        f.write(f"  Summary: {len(passed)} passed, {len(failed)} failed\n\n")


def write_results_by_placement(f, results, title):
    """
    Write summarized results organized by op name.
    Collapses data patterns - a placement combo passes only if ALL pass.
    """
    by_op = _group_results_by_op(results)

    f.write(f"\n{'=' * 80}\n")
    f.write(f"{title} (SUMMARIZED)\n")
    f.write(f"{'=' * 80}\n\n")

    for op_name in sorted(by_op.keys()):
        f.write(f"{op_name}:\n")
        passed, failed = [], []

        for base_placements, results_list in sorted(by_op[op_name].items()):
            placement_str = ", ".join(str(p) for p in base_placements)
            all_passed = all(success for _, success, _ in results_list)

            if all_passed:
                passed.append(f"  [PASS] {placement_str}")
            else:
                failed.append(f"  [FAIL] {placement_str}")

        for line in passed + failed:
            f.write(line + "\n")
        f.write(f"  Summary: {len(passed)} passed, {len(failed)} failed\n\n")


def write_results(f, results, title):
    """Write results using the appropriate mode based on SUMMARIZE_MODE flag."""
    if SUMMARIZE_MODE:
        write_results_by_placement(f, results, title)
    else:
        write_results_by_op(f, results, title)


# =============================================================================
# Test Class
# =============================================================================


class TestPartialPropagation(DTensorOpTestBase):
    """
    Tests that propagating partials through pointwise ops gives the same
    result as redistributing all inputs to Replicate first.
    """

    def _compare_propagate_vs_redistribute(
        self,
        op: Callable,
        aten_op: OpOverload,
        inputs: list[DTensor],
        mesh: DeviceMesh,
    ) -> tuple[bool, str]:
        """
        Compare two approaches:
        1. Propagate partials through op (with experimental strategy), then
           redistribute result to Replicate
        2. Redistribute all inputs to Replicate first, then apply op

        Returns (success, error_message)
        """
        # Baseline: Redistribute first, then compute
        inputs_replicate = [inp.redistribute(mesh, [Replicate()]) for inp in inputs]
        result2 = op(*inputs_replicate)
        result2_full = result2.to_local()

        # Experimental: Override strategy, propagate partial, then redistribute
        with override_op_strategies([aten_op]):
            result1 = op(*inputs)
            result1_full = result1.redistribute(mesh, [Replicate()]).to_local()

        try:
            torch.testing.assert_close(result1_full, result2_full)
            return True, ""
        except AssertionError as e:
            return False, f"Results differ: {str(e)}"

    def _run_unary_test(
        self,
        torch_op: Callable,
        aten_op: OpOverload,
        partial_cls,
        partial_arg,
        partial_name: str,
        data: torch.Tensor,
        pattern_name: str,
        mesh: DeviceMesh,
    ) -> tuple[tuple, tuple[bool, str]]:
        """Run a single unary op test and return (key, result)."""
        dt = DTensor.from_local(data, mesh, [partial_cls(partial_arg)])
        success, error = self._compare_propagate_vs_redistribute(
            torch_op, aten_op, [dt], mesh
        )
        key = (torch_op.__name__, partial_name, pattern_name)
        return key, (success, error)

    def _run_binary_test(
        self,
        torch_op: Callable,
        aten_op: OpOverload,
        dt1: DTensor,
        dt2: DTensor,
        placement1_name: str,
        placement2_name: str,
        pattern_name: str,
        mesh: DeviceMesh,
    ) -> tuple[tuple, tuple[bool, str]]:
        """Run a single binary op test and return (key, result)."""
        success, error = self._compare_propagate_vs_redistribute(
            torch_op, aten_op, [dt1, dt2], mesh
        )
        key = (torch_op.__name__, placement1_name, placement2_name, pattern_name)
        return key, (success, error)

    @with_comms
    def test_unary_ops(self):
        """Test all unary ops with all partial types using three data patterns."""
        mesh = self.build_device_mesh()
        results = {}
        data = generate_test_data(self.device_type, dist.get_rank())

        test_patterns = [
            ("mono", data["mono1"]),
            ("varied", data["varied1"]),
            ("allneg", data["neg1"]),
        ]

        for partial_cls, partial_arg, partial_name in PARTIAL_TYPES:
            for torch_op, aten_op in UNARY_OPS:
                for pattern_name, tensor in test_patterns:
                    key, result = self._run_unary_test(
                        torch_op,
                        aten_op,
                        partial_cls,
                        partial_arg,
                        partial_name,
                        tensor,
                        pattern_name,
                        mesh,
                    )
                    results[key] = result

        if dist.get_rank() == 0:
            with open(OUTPUT_FILE, "a") as f:
                write_results(f, results, "UNARY OPS")

    @with_comms
    def test_binary_ops(self):
        """
        Test binary ops with:
        1. Matching partial types (P + P)
        2. Partial + Replicate combinations (P + R only)
        """
        mesh = self.build_device_mesh()
        results = {}
        rank = dist.get_rank()

        data = generate_test_data(self.device_type, rank)
        rep_data = generate_replicate_data(self.device_type)

        for partial_cls, partial_arg, partial_name in PARTIAL_TYPES:
            for torch_op, aten_op in BINARY_OPS:
                # Define all test cases: (pattern_name, data1, data2, is_p_plus_r)
                test_cases = [
                    # P + P tests
                    ("mono", data["mono1"], data["mono2"], False),
                    ("varied", data["varied1"], data["varied2"], False),
                    ("allneg", data["neg1"], data["neg2"], False),
                    # P + R tests
                    ("mono", data["mono1"], rep_data["pos"], True),
                    ("varied", data["varied1"], rep_data["pos"], True),
                    ("allneg", data["neg1"], rep_data["neg"], True),
                    ("rneg", data["mono1"], rep_data["neg"], True),
                ]

                for pattern_name, tensor1, tensor2, is_p_plus_r in test_cases:
                    dt1 = DTensor.from_local(tensor1, mesh, [partial_cls(partial_arg)])
                    if is_p_plus_r:
                        dt2 = DTensor.from_local(
                            tensor2, mesh, [Replicate()], run_check=True
                        )
                        placement2_name = "R"
                    else:
                        dt2 = DTensor.from_local(
                            tensor2, mesh, [partial_cls(partial_arg)]
                        )
                        placement2_name = partial_name

                    key, result = self._run_binary_test(
                        torch_op,
                        aten_op,
                        dt1,
                        dt2,
                        partial_name,
                        placement2_name,
                        pattern_name,
                        mesh,
                    )
                    results[key] = result

        if dist.get_rank() == 0:
            with open(OUTPUT_FILE, "a") as f:
                write_results(f, results, "BINARY OPS")


if __name__ == "__main__":
    # Clear output file at start (only rank 0)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        open(OUTPUT_FILE, "w").close()

    run_tests()
