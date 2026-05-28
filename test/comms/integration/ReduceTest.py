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


class ReduceTest(unittest.TestCase):
    """Test class for reduce operations in TorchComm."""

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
        # Create all test parameters
        normal_test_cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        # Max ranks for int8 is 15 because sum(1..16) = 136 which overflows int8 (>127)
        return filter_int8_overflow_cases(normal_test_cases, self.num_ranks, 15)

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

    def _sync_reduce(self, count, dtype, op):
        """Test synchronous reduce with work object."""
        print(
            f"Testing sync reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Everyone creates the input tensor
        tensor = self._create_input_tensor(count, dtype)

        # Call reduce
        work = self.torchcomm.reduce(tensor, root_rank, op, False)
        work.wait()

        # Verify the results
        self._verify_results(tensor, op, root_rank)

    def _sync_reduce_no_work(self, count, dtype, op):
        """Test synchronous reduce without work object."""
        print(
            f"Testing sync reduce without work object with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Everyone creates the input tensor
        tensor = self._create_input_tensor(count, dtype)

        # Call reduce without keeping the work object
        self.torchcomm.reduce(tensor, root_rank, op, False)

        # Verify the results
        self._verify_results(tensor, op, root_rank)

    def _async_reduce(self, count, dtype, op):
        """Test asynchronous reduce with wait."""
        print(
            f"Testing async reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Everyone creates the input tensor
        tensor = self._create_input_tensor(count, dtype)

        # Call reduce
        work = self.torchcomm.reduce(tensor, root_rank, op, True)

        # Wait for the reduce to complete
        work.wait()

        # Verify the results
        self._verify_results(tensor, op, root_rank)

    def _async_reduce_early_reset(self, count, dtype, op):
        """Test asynchronous reduce with early reset."""
        print(
            f"Testing async reduce with early reset with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Everyone creates the input tensor
        tensor = self._create_input_tensor(count, dtype)

        # Call reduce
        work = self.torchcomm.reduce(tensor, root_rank, op, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(tensor, op, root_rank)

    def _reduce_input_deleted(self, count, dtype, op):
        """Test asynchronous reduce with input deleted after enqueue."""
        print(
            f"Testing async reduce with input deleted after enqueue with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Create input tensor and enqueue operation
        tensor = self._create_input_tensor(count, dtype)

        # Call reduce with async_op = False
        self.torchcomm.reduce(tensor, root_rank, op, False)

        # Delete the tensor to simulate it going out of scope
        del tensor

        # Note: For reduce, the operation is in-place, so we need to create a new tensor
        # to verify results since the original was deleted. This test primarily validates
        # that the operation completes without crashing when input is deleted.

    def _graph_reduce(self, count, dtype, op):
        """Test CUDA Graph reduce."""
        print(
            f"Testing CUDA Graph reduce with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            root_rank = 0

            # Create input tensor AFTER setting non-default stream but BEFORE graph capture
            tensor = self._create_input_tensor(count, dtype)
            original_values = tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the reset + reduce operations in the graph
            with torch.cuda.graph(graph):
                # Call reduce without keeping the work object
                self.torchcomm.reduce(tensor, root_rank, op, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset tensor before graph replay
                tensor.copy_(original_values)

                graph.replay()

                # Verify the results on root rank after each replay
                self._verify_results(tensor, op, root_rank)

    def _graph_reduce_input_deleted(self, count, dtype, op):
        """Test CUDA Graph reduce with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph reduce with input deleted after graph creation with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        root_rank = 0

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensor for graph capture
            tensor = self._create_input_tensor(count, dtype)

            # Capture the reset + reduce operations in the graph
            with torch.cuda.graph(graph):
                # Call reduce without keeping the work object
                self.torchcomm.reduce(tensor, root_rank, op, False)

            # Delete the tensor to simulate it going out of scope
            del tensor

            # Replay the captured graph multiple times even though tensor is deleted
            for _ in range(self.num_replays):
                graph.replay()

                # Note: For reduce with deleted input, we can't verify results
                # since the tensor was deleted. This test primarily validates
                # that the graph replay completes without crashing.

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if dtype == torch.float or dtype == torch.bfloat16:
            return torch.ones(count, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(self.rank + 1)
        return None

    def _calculate_expected_result(self, op):
        """Calculate expected result based on operation."""
        if op == ReduceOp.SUM:
            return self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            return self.num_ranks
        elif op == ReduceOp.AVG:
            return (self.num_ranks * (self.num_ranks + 1) / 2) / self.num_ranks
        else:
            raise RuntimeError("Unsupported reduce operation")

    def synchronize_stream(self):
        """Synchronize the current stream."""
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        # For CPU devices, no synchronization needed

    def _verify_results(self, output_tensor, op, root_rank):
        """Verify the results of the reduce operation."""
        if self.rank != root_rank:
            self.synchronize_stream()
            return

        # Calculate expected result
        expected = self._calculate_expected_result(op)

        # Compare output with expected tensor
        description = f"reduce with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def test_sync_reduce(self):
        """Test synchronous reduce with work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_reduce(count, dtype, op)

    def test_sync_reduce_no_work(self):
        """Test synchronous reduce without work object."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_reduce_no_work(count, dtype, op)

    def test_async_reduce(self):
        """Test asynchronous reduce with wait."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_reduce(count, dtype, op)

    def test_async_reduce_early_reset(self):
        """Test asynchronous reduce with early reset."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._async_reduce_early_reset(count, dtype, op)

    def test_reduce_input_deleted(self):
        """Test asynchronous reduce with input deleted after enqueue."""
        for count, dtype, op in self.get_test_cases():
            with self.subTest(count=count, dtype=dtype, op=op):
                self._reduce_input_deleted(count, dtype, op)


if __name__ == "__main__":
    unittest.main()
