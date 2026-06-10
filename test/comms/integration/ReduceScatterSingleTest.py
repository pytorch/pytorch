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
from torch.comms import ReduceOp


class ReduceScatterSingleTest(unittest.TestCase):
    """Test class for reduce_scatter_single operations in TorchComm."""

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
        normal_test_cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        # Max ranks for int8 is 11 because 12 * 12 = 144 overflows int8 (>127)
        return filter_int8_overflow_cases(normal_test_cases, self.num_ranks, 11)

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

    def _sync_reduce_scatter_single(self, count, dtype, op):
        """Test synchronous reduce_scatter_single with work object."""
        print(
            f"Testing sync reduce_scatter_single with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call reduce_scatter_single
        work = self.torchcomm.reduce_scatter_single(
            output_tensor, input_tensor, op, False
        )
        work.wait()

        # Verify the results
        self._verify_results(output_tensor, op)

    def _sync_reduce_scatter_single_no_work(self, count, dtype, op):
        """Test synchronous reduce_scatter_single without work object."""
        print(
            f"Testing sync reduce_scatter_single without work object with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call reduce_scatter_single without keeping the work object
        self.torchcomm.reduce_scatter_single(output_tensor, input_tensor, op, False)

        # Verify the results
        self._verify_results(output_tensor, op)

    def _async_reduce_scatter_single(self, count, dtype, op):
        """Test asynchronous reduce_scatter_single with wait."""
        print(
            f"Testing async reduce_scatter_single with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call reduce_scatter_single
        work = self.torchcomm.reduce_scatter_single(
            output_tensor, input_tensor, op, True
        )

        # Wait for the reduce_scatter_single to complete
        work.wait()

        # Verify the results
        self._verify_results(output_tensor, op)

    def _async_reduce_scatter_single_early_reset(self, count, dtype, op):
        """Test asynchronous reduce_scatter_single with early reset."""
        print(
            f"Testing async reduce_scatter_single with early reset with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call reduce_scatter_single
        work = self.torchcomm.reduce_scatter_single(
            output_tensor, input_tensor, op, True
        )

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output_tensor, op)

    def _reduce_scatter_single_input_deleted(self, count, dtype, op):
        """Test asynchronous reduce_scatter_single with input deleted after enqueue."""
        print(
            f"Testing async reduce_scatter_single with input deleted after enqueue with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create output tensor that persists throughout the test
        output_tensor = self._create_output_tensor(count, dtype)

        # Create input tensor and enqueue operation
        input_tensor = self._create_input_tensor(count, dtype)

        # Call reduce_scatter_single with async_op = False
        self.torchcomm.reduce_scatter_single(output_tensor, input_tensor, op, False)

        # Delete the input tensor to simulate it going out of scope
        del input_tensor

        # Verify the results
        self._verify_results(output_tensor, op)

    def _graph_reduce_scatter_single(self, count, dtype, op):
        """Test CUDA Graph reduce_scatter_single."""
        print(
            f"Testing CUDA Graph reduce_scatter_single with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input and output tensors AFTER setting non-default stream but BEFORE graph capture
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensor = self._create_output_tensor(count, dtype)
            original_output_tensor = output_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the reduce_scatter_single operation in the graph
            with torch.cuda.graph(graph):
                # Call reduce_scatter_single without keeping the work object
                self.torchcomm.reduce_scatter_single(
                    output_tensor, input_tensor, op, False
                )

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output buffer before graph replay
                output_tensor.copy_(original_output_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensor, op)

    def _graph_reduce_scatter_single_input_deleted(self, count, dtype, op):
        """Test CUDA Graph reduce_scatter_single with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph reduce_scatter_single with input deleted after graph creation with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create output tensor that persists throughout the test
            output_tensor = self._create_output_tensor(count, dtype)
            original_output_tensor = output_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensor in a limited scope
            input_tensor = self._create_input_tensor(count, dtype)

            # Capture the reduce_scatter_single operation in the graph
            with torch.cuda.graph(graph):
                # Call reduce_scatter_single without keeping the work object
                self.torchcomm.reduce_scatter_single(
                    output_tensor, input_tensor, op, False
                )

            # Input tensor goes out of scope here and gets deleted
            del input_tensor

        # Replay the captured graph multiple times even though input is deleted
        for _ in range(self.num_replays):
            # Reset output buffer before graph replay
            output_tensor.copy_(original_output_tensor)

            graph.replay()

            # Verify the results after each replay
            self._verify_results(output_tensor, op)

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        input_tensor = torch.zeros(count * self.num_ranks, **options)

        # Create a tensor of rank values [1, 2, ..., num_ranks]
        ranks = torch.arange(1, self.num_ranks + 1, **options)

        # For each rank, fill its section with its rank value
        for r in range(self.num_ranks):
            # Use slice operation to get the section for this rank
            section = input_tensor[r * count : (r + 1) * count]
            # Fill the entire section with the rank value in one operation
            section.fill_(ranks[r].item())

        return input_tensor

    def _create_output_tensor(self, count, dtype):
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count, **options)

    def _verify_results(self, output_tensor, op):
        """Verify the results of the reduce_scatter_single operation."""
        # Calculate expected value based on operation type
        expected_value = 0
        if op == ReduceOp.SUM:
            # Sum: num_ranks * (rank+1)
            expected_value = self.num_ranks * (self.rank + 1)
        elif op == ReduceOp.MAX:
            # Max: rank+1
            expected_value = self.rank + 1
        elif op == ReduceOp.AVG:
            # Avg: (num_ranks * (rank+1)) / num_ranks = rank+1
            expected_value = self.rank + 1

        # Compare output with expected tensor
        description = f"reduce_scatter_single with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(
                output_tensor.cpu(), float(expected_value)
            )
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected_value)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def test_sync_reduce_scatter_single(self):
        """Test synchronous reduce_scatter_single with work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_reduce_scatter_single(count, dtype, op)

    def test_sync_reduce_scatter_single_no_work(self):
        """Test synchronous reduce_scatter_single without work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_reduce_scatter_single_no_work(count, dtype, op)

    def test_async_reduce_scatter_single(self):
        """Test asynchronous reduce_scatter_single with wait."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_reduce_scatter_single(count, dtype, op)

    def test_async_reduce_scatter_single_early_reset(self):
        """Test asynchronous reduce_scatter_single with early reset."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_reduce_scatter_single_early_reset(count, dtype, op)

    def test_reduce_scatter_single_input_deleted(self):
        """Test asynchronous reduce_scatter_single with input deleted after enqueue."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._reduce_scatter_single_input_deleted(count, dtype, op)


if __name__ == "__main__":
    unittest.main()
