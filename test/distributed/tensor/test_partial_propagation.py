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


def write_results_by_op(f, results, title):
    """
    Write results organized by op name.
    Collapses mono/varied/allneg - a placement combo passes only if ALL pass.
    Shows which data patterns failed for failed cases.
    """
    # Group results by op, then by placement (without mono/varied/allneg)
    by_op = defaultdict(lambda: defaultdict(list))
    for key, (success, error) in results.items():
        op_name = key[0]
        placements = key[1:]
        # Remove the last element if it's a data pattern type
        if placements and placements[-1] in ("mono", "varied", "allneg", "rneg"):
            base_placements = placements[:-1]
            data_type = placements[-1]
        else:
            base_placements = placements
            data_type = None
        by_op[op_name][base_placements].append((data_type, success, error))

    f.write(f"\n{'=' * 80}\n")
    f.write(f"{title}\n")
    f.write(f"{'=' * 80}\n\n")

    # Write results for each op
    for op_name in sorted(by_op.keys()):
        f.write(f"{op_name}:\n")

        passed = []
        failed = []

        for base_placements, results_list in sorted(by_op[op_name].items()):
            placement_str = ", ".join(str(p) for p in base_placements)
            # A combo passes only if ALL (mono, varied, allneg) pass
            all_passed = all(success for _, success, _ in results_list)

            if all_passed:
                passed.append(f"  [PASS] {placement_str}")
            else:
                # Show which data patterns failed
                failed_patterns = [
                    data_type for data_type, success, _ in results_list if not success
                ]
                failed_str = (
                    ", ".join(failed_patterns) if failed_patterns else "unknown"
                )
                failed.append(f"  [FAIL] {placement_str} (failed: {failed_str})")

        # Write passed tests first
        for line in passed:
            f.write(line + "\n")

        # Then failed tests
        for line in failed:
            f.write(line + "\n")

        f.write(f"  Summary: {len(passed)} passed, {len(failed)} failed\n\n")


def write_results_by_placement(f, results, title):
    """
    Write summarized results organized by op name.
    Collapses mono/varied/allneg - a placement combo passes only if ALL pass.
    """
    # Group results by op, then by placement (without mono/varied/allneg)
    by_op = defaultdict(lambda: defaultdict(list))
    for key, (success, error) in results.items():
        op_name = key[0]
        placements = key[1:]
        # Remove the last element if it's a data pattern type
        if placements and placements[-1] in ("mono", "varied", "allneg", "rneg"):
            base_placements = placements[:-1]
            data_type = placements[-1]
        else:
            base_placements = placements
            data_type = None
        by_op[op_name][base_placements].append((data_type, success, error))

    f.write(f"\n{'=' * 80}\n")
    f.write(f"{title} (SUMMARIZED)\n")
    f.write(f"{'=' * 80}\n\n")

    # Write results for each op
    for op_name in sorted(by_op.keys()):
        f.write(f"{op_name}:\n")

        passed = []
        failed = []

        for base_placements, results_list in sorted(by_op[op_name].items()):
            placement_str = ", ".join(str(p) for p in base_placements)
            # A combo passes only if ALL (mono and varied) pass
            all_passed = all(success for _, success, _ in results_list)

            if all_passed:
                passed.append(f"  [PASS] {placement_str}")
            else:
                failed.append(f"  [FAIL] {placement_str}")

        # Write passed tests first
        for line in passed:
            f.write(line + "\n")

        # Then failed tests
        for line in failed:
            f.write(line + "\n")

        f.write(f"  Summary: {len(passed)} passed, {len(failed)} failed\n\n")


def write_results(f, results, title):
    """Write results using the appropriate mode based on SUMMARIZE_MODE flag."""
    if SUMMARIZE_MODE:
        write_results_by_placement(f, results, title)
    else:
        write_results_by_op(f, results, title)


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

        Note: We don't catch exceptions during collective ops (redistribute)
        to avoid thread desync in multi-threaded tests.
        """
        # Method 2 first (baseline): Redistribute first, then compute
        # Do this BEFORE overriding strategies so we use normal behavior
        inputs_replicate = [inp.redistribute(mesh, [Replicate()]) for inp in inputs]
        result2 = op(*inputs_replicate)
        result2_full = result2.to_local()

        # Method 1: Override strategy, propagate partial, then redistribute
        with override_op_strategies([aten_op]):
            result1 = op(*inputs)
            result1_full = result1.redistribute(mesh, [Replicate()]).to_local()

        # Compare results - this is the only place we catch exceptions
        try:
            torch.testing.assert_close(result1_full, result2_full)
            return True, ""
        except AssertionError as e:
            return False, f"Results differ: {str(e)}"

    @with_comms
    def test_unary_ops(self):
        """Test all unary ops with all partial types using three data patterns."""
        mesh = self.build_device_mesh()
        results = {}
        rank = dist.get_rank()

        for partial_cls, partial_arg, partial_name in PARTIAL_TYPES:
            for torch_op, aten_op in UNARY_OPS:
                # Test 1: Monotonic data
                local_mono = (
                    torch.tensor([[-1.0, -2.0], [-3.0, 4.0]], device=self.device_type)
                    + rank
                )
                dt = DTensor.from_local(local_mono, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt], mesh
                )
                results[(torch_op.__name__, partial_name, "mono")] = (
                    success,
                    error,
                )

                # Test 2: Varied data
                if rank % 2 == 0:
                    local_var = torch.tensor(
                        [[10.0, 1.0], [1.0, 10.0]], device=self.device_type
                    )
                else:
                    local_var = torch.tensor(
                        [[1.0, 10.0], [10.0, 1.0]], device=self.device_type
                    )
                dt = DTensor.from_local(local_var, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt], mesh
                )
                results[(torch_op.__name__, partial_name, "varied")] = (
                    success,
                    error,
                )

                # Test 3: All negative values
                local_neg = (
                    torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device=self.device_type)
                    - rank
                )
                dt = DTensor.from_local(local_neg, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt], mesh
                )
                results[(torch_op.__name__, partial_name, "allneg")] = (
                    success,
                    error,
                )

        # Write to file (only rank 0)
        if dist.get_rank() == 0:
            with open(OUTPUT_FILE, "a") as f:
                write_results(f, results, "UNARY OPS")

    @with_comms
    def test_binary_ops(self):
        """
        Test binary ops with:
        1. Matching partial types (P + P)
        2. Partial + Replicate combinations (P + R only, not R + P)
        """
        mesh = self.build_device_mesh()
        results = {}
        rank = dist.get_rank()

        # Fixed data for Replicate inputs (same across all ranks)
        replicate_data = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=self.device_type)
        replicate_data_neg = torch.tensor(
            [[-2.0, -3.0], [-4.0, -5.0]], device=self.device_type
        )

        for partial_cls, partial_arg, partial_name in PARTIAL_TYPES:
            for torch_op, aten_op in BINARY_OPS:
                # Test 1: Monotonic data (lower ranks have smaller values)
                local1_mono = (
                    torch.tensor([[-1.0, -2.0], [-3.0, 4.0]], device=self.device_type)
                    + rank
                )
                local2_mono = (
                    torch.tensor([[-5.0, 6.0], [-7.0, -8.0]], device=self.device_type)
                    + rank
                )

                # P + P (monotonic)
                dt1 = DTensor.from_local(local1_mono, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(local2_mono, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, partial_name, "mono")] = (
                    success,
                    error,
                )

                # P + R (monotonic) - use fixed replicate data
                dt1 = DTensor.from_local(local1_mono, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(
                    replicate_data, mesh, [Replicate()], run_check=True
                )
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, "R", "mono")] = (
                    success,
                    error,
                )

                # Test 2: Varied data (min/max on different ranks for different elements)
                if rank % 2 == 0:
                    local1_var = torch.tensor(
                        [[10.0, 1.0], [1.0, 10.0]], device=self.device_type
                    )
                    local2_var = torch.tensor(
                        [[1.0, 10.0], [10.0, 1.0]], device=self.device_type
                    )
                else:
                    local1_var = torch.tensor(
                        [[1.0, 10.0], [10.0, 1.0]], device=self.device_type
                    )
                    local2_var = torch.tensor(
                        [[10.0, 1.0], [1.0, 10.0]], device=self.device_type
                    )

                # P + P (varied)
                dt1 = DTensor.from_local(local1_var, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(local2_var, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, partial_name, "varied")] = (
                    success,
                    error,
                )

                # P + R (varied) - use fixed replicate data
                dt1 = DTensor.from_local(local1_var, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(
                    replicate_data, mesh, [Replicate()], run_check=True
                )
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, "R", "varied")] = (
                    success,
                    error,
                )

                # Test 3: All negative values
                local1_neg = (
                    torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device=self.device_type)
                    - rank
                )
                local2_neg = (
                    torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device=self.device_type)
                    - rank
                )

                # P + P (all negative)
                dt1 = DTensor.from_local(local1_neg, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(local2_neg, mesh, [partial_cls(partial_arg)])
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, partial_name, "allneg")] = (
                    success,
                    error,
                )

                # P + R (all negative) - use fixed negative replicate data
                dt1 = DTensor.from_local(local1_neg, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(
                    replicate_data_neg, mesh, [Replicate()], run_check=True
                )
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, "R", "allneg")] = (
                    success,
                    error,
                )

                # Test 4: P + R with negative replicate (partial has mixed, replicate is negative)
                dt1 = DTensor.from_local(local1_mono, mesh, [partial_cls(partial_arg)])
                dt2 = DTensor.from_local(
                    replicate_data_neg, mesh, [Replicate()], run_check=True
                )
                success, error = self._compare_propagate_vs_redistribute(
                    torch_op, aten_op, [dt1, dt2], mesh
                )
                results[(torch_op.__name__, partial_name, "R", "rneg")] = (
                    success,
                    error,
                )

        # Write to file (only rank 0)
        if dist.get_rank() == 0:
            with open(OUTPUT_FILE, "a") as f:
                write_results(f, results, "BINARY OPS")


if __name__ == "__main__":
    # Clear output file at start (only rank 0)
    # Use RANK env var since dist isn't initialized yet when running with torchrun
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        open(OUTPUT_FILE, "w").close()

    run_tests()
