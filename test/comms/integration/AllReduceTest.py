#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

from integration.helpers.TorchCommTestHelpers import (
    filter_int8_overflow_cases,
    get_dtype_name,
    get_op_name,
    is_full_sweep,
    TorchCommTestWrapper,
)

import torch
from torch.comms import RedOpType, ReduceOp


class AllReduceTest(unittest.TestCase):
    """Test class for all_reduce operations in TorchComm."""

    # Class variables for test parameters
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    ops = (
        [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.AVG]
        if is_full_sweep()
        else [ReduceOp.SUM]
    )
    num_replays = 4

    def get_test_cases(self):
        # Create a test params
        normal_test_cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        premul_sum_types_ops = [
            (torch.half, ReduceOp.PREMUL_SUM(2.0)),
            (torch.float, ReduceOp.PREMUL_SUM(2.0)),
            (torch.double, ReduceOp.PREMUL_SUM(2.0)),
            (
                torch.bfloat16,
                ReduceOp.PREMUL_SUM(
                    torch.ones(1, dtype=torch.bfloat16, device=self.device) * 2.0
                ),
            ),
        ]
        premul_sum_test_cases = [
            (count, dtype, op)
            for count, (dtype, op) in itertools.product(
                self.counts, premul_sum_types_ops
            )
        ]
        # Max ranks for int8 is 15 because sum(1..16) = 136 which overflows int8 (>127)
        return (
            filter_int8_overflow_cases(normal_test_cases, self.num_ranks, 15)
            + premul_sum_test_cases
        )

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _sync_all_reduce(self, count, dtype, op):
        """Test synchronous all_reduce with work object."""
        print(
            f"Testing sync all_reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input tensor with rank-specific values
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_reduce
        work = self.torchcomm.all_reduce(input_tensor, op, False)
        work.wait()

        # Verify the results
        self._verify_results(input_tensor, op)

    def _sync_all_reduce_no_work(self, count, dtype, op):
        """Test synchronous all_reduce without work object."""
        print(
            f"Testing sync all_reduce without work object with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input tensor with rank-specific values
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_reduce without keeping the work object
        self.torchcomm.all_reduce(input_tensor, op, False)

        # Verify the results
        self._verify_results(input_tensor, op)

    def _async_all_reduce(self, count, dtype, op):
        """Test asynchronous all_reduce with wait."""
        print(
            f"Testing async all_reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input tensor with rank-specific values
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_reduce
        work = self.torchcomm.all_reduce(input_tensor, op, True)

        # Wait for the all_reduce to complete
        work.wait()

        # Verify the results
        self._verify_results(input_tensor, op)

    def _async_all_reduce_early_reset(self, count, dtype, op):
        """Test asynchronous all_reduce with early reset."""
        print(
            f"Testing async all_reduce with early reset with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input tensor with rank-specific values
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_reduce
        work = self.torchcomm.all_reduce(input_tensor, op, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(input_tensor, op)

    def _all_reduce_input_deleted(self, count, dtype, op):
        """Test asynchronous all_reduce with input deleted after enqueue."""
        print(
            f"Testing async all_reduce with input deleted after enqueue with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input tensor and enqueue operation
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_reduce with async_op = False
        self.torchcomm.all_reduce(input_tensor, op, False)

        # Delete the input tensor to simulate it going out of scope
        del input_tensor

        # Note: For all_reduce, the operation is in-place, so we need to create a new tensor
        # to verify results since the original was deleted. This test primarily validates
        # that the operation completes without crashing when input is deleted.

    def _graph_all_reduce(self, count, dtype, op):
        """Test CUDA Graph all_reduce."""
        print(
            f"Testing CUDA Graph all_reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input tensor AFTER setting non-default stream but BEFORE graph capture
            input_tensor = self._create_input_tensor(count, dtype)
            original_values = input_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the reset + all_reduce operations in the graph
            with torch.cuda.graph(graph):
                # Call all_reduce without keeping the work object
                self.torchcomm.all_reduce(input_tensor, op, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset input tensor before graph replay
                input_tensor.copy_(original_values)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(input_tensor, op)

    def _graph_all_reduce_input_deleted(self, count, dtype, op):
        """Test CUDA Graph all_reduce with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph all_reduce with input deleted after graph creation with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensor for graph capture
            input_tensor = self._create_input_tensor(count, dtype)

            # Capture the all_reduce operation in the graph
            with torch.cuda.graph(graph):
                # Call all_reduce without keeping the work object
                self.torchcomm.all_reduce(input_tensor, op, False)

            # Delete the input tensor to simulate it going out of scope
            del input_tensor

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                graph.replay()

                # Note: For all_reduce with deleted input, we can't verify results
                # since the tensor was deleted. This test primarily validates
                # that the graph replay completes without crashing.

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if (
            dtype == torch.float
            or dtype == torch.bfloat16
            or dtype == torch.double
            or dtype == torch.half
        ):
            return torch.ones(count, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(self.rank + 1)
        return None

    def _calculate_expected_result(self, op):
        """Calculate expected result based on operation."""
        if op.type == RedOpType.SUM:
            # Sum: sum of all ranks (1+2+...+num_ranks)
            return self.num_ranks * (self.num_ranks + 1) // 2
        elif op.type == RedOpType.MAX:
            # Max: highest rank value (num_ranks)
            return self.num_ranks
        elif op.type == RedOpType.AVG:
            # Avg: average of all ranks
            return (self.num_ranks * (self.num_ranks + 1) / 2) / self.num_ranks
        elif op.type == RedOpType.PREMUL_SUM:
            # PremulSum: sum of all ranks multiplied by 2.0
            return self.num_ranks * (self.num_ranks + 1)
        else:
            raise RuntimeError("Unsupported reduce operation")

    def _verify_results(self, input_tensor, op):
        """Verify the results of the all_reduce operation."""
        # Calculate expected result
        expected = self._calculate_expected_result(op)

        # Compare input with expected tensor

        def msg(error: str) -> str:
            return f"all_reduce with op {get_op_name(op)}: {error}"

        # Create expected tensor with the same size and dtype as input
        if input_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(input_tensor.cpu(), float(expected))
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor, msg=msg)

        else:
            expected_tensor = torch.full_like(input_tensor.cpu(), expected)
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor, msg=msg)

    def test_sync_all_reduce(self):
        """Test synchronous all_reduce with work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_all_reduce(count, dtype, op)

    def test_sync_all_reduce_no_work(self):
        """Test synchronous all_reduce without work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_all_reduce_no_work(count, dtype, op)

    def test_async_all_reduce(self):
        """Test asynchronous all_reduce with wait."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_all_reduce(count, dtype, op)

    def test_async_all_reduce_early_reset(self):
        """Test asynchronous all_reduce with early reset."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_all_reduce_early_reset(count, dtype, op)

    def test_all_reduce_input_deleted(self):
        """Test asynchronous all_reduce with input deleted after enqueue."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._all_reduce_input_deleted(count, dtype, op)


if __name__ == "__main__":
    unittest.main()
