#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    is_full_sweep,
    TorchCommTestWrapper,
)

import torch


class ScatterTest(unittest.TestCase):
    """Test class for scatter operations in TorchComm."""

    # Class variables for test parameters
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    num_replays = 4

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

    def _sync_scatter(self, count, dtype):
        """Test synchronous scatter with work object."""
        print(
            f"Testing sync scatter with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Root rank will send data to all ranks
        root_rank = 0

        # Only the root rank needs to create input tensors
        inputs = []
        if self.rank == root_rank:
            inputs = self._create_input_tensors(count, dtype)

        # Create output tensor to receive data
        output = self._create_output_tensor(count, dtype)

        # Call scatter
        work = self.torchcomm.scatter(output, inputs, root_rank, False)
        work.wait()

        # Verify the results
        self._verify_results(output)

    def _sync_scatter_no_work(self, count, dtype):
        """Test synchronous scatter without work object."""
        print(
            f"Testing sync scatter without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Root rank will send data to all ranks
        root_rank = 0

        # Only the root rank needs to create input tensors
        inputs = []
        if self.rank == root_rank:
            inputs = self._create_input_tensors(count, dtype)

        # Create output tensor to receive data
        output = self._create_output_tensor(count, dtype)

        # Call scatter without keeping the work object
        self.torchcomm.scatter(output, inputs, root_rank, False)

        # Verify the results
        self._verify_results(output)

    def _async_scatter(self, count, dtype):
        """Test asynchronous scatter with wait."""
        print(
            f"Testing async scatter with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Root rank will send data to all ranks
        root_rank = 0

        # Only the root rank needs to create input tensors
        inputs = []
        if self.rank == root_rank:
            inputs = self._create_input_tensors(count, dtype)

        # Create output tensor to receive data
        output = self._create_output_tensor(count, dtype)

        # Call scatter
        work = self.torchcomm.scatter(output, inputs, root_rank, True)

        # Wait for the scatter to complete
        work.wait()

        # Verify the results
        self._verify_results(output)

    def _async_scatter_early_reset(self, count, dtype):
        """Test asynchronous scatter with early reset."""
        print(
            f"Testing async scatter with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Root rank will send data to all ranks
        root_rank = 0

        # Only the root rank needs to create input tensors
        inputs = []
        if self.rank == root_rank:
            inputs = self._create_input_tensors(count, dtype)

        # Create output tensor to receive data
        output = self._create_output_tensor(count, dtype)

        # Call scatter
        work = self.torchcomm.scatter(output, inputs, root_rank, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output)

    def _scatter_input_deleted(self, count, dtype):
        """Test asynchronous scatter with input deleted after enqueue."""
        print(
            f"Testing async scatter with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0

        # Create output tensor that persists throughout the test
        output = self._create_output_tensor(count, dtype)

        # Create input tensors and enqueue operation
        inputs = []
        if self.rank == root_rank:
            inputs = self._create_input_tensors(count, dtype)

        # Call scatter with async_op = False
        self.torchcomm.scatter(output, inputs, root_rank, False)

        # Delete the input tensors to simulate them going out of scope
        if self.rank == root_rank:
            del inputs

        # Verify the results
        self._verify_results(output)

    def _graph_scatter(self, count, dtype):
        """Test CUDA Graph scatter."""
        print(
            f"Testing CUDA Graph scatter with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input and output tensors AFTER setting non-default stream but BEFORE graph capture
            root_rank = 0
            inputs = []
            if self.rank == root_rank:
                inputs = self._create_input_tensors(count, dtype)
            output = self._create_output_tensor(count, dtype)
            original_output = output.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the scatter operation in the graph
            with torch.cuda.graph(graph):
                # Call scatter without keeping the work object
                self.torchcomm.scatter(output, inputs, root_rank, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output buffer before each replay
                output.copy_(original_output)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output)

    def _graph_scatter_input_deleted(self, count, dtype):
        """Test CUDA Graph scatter with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph scatter with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Root rank will send data to all ranks
            root_rank = 0

            # Create output tensor that persists throughout the test
            output = self._create_output_tensor(count, dtype)
            original_output = output.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensors for graph capture
            inputs = []
            if self.rank == root_rank:
                inputs = self._create_input_tensors(count, dtype)

            # Capture the scatter operation in the graph
            with torch.cuda.graph(graph):
                # Call scatter without keeping the work object
                self.torchcomm.scatter(output, inputs, root_rank, False)

            # Delete the input tensors to simulate them going out of scope
            if self.rank == root_rank:
                del inputs

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                # Reset output buffer before each replay
                output.copy_(original_output)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output)

    def _create_input_tensors(self, count, dtype):
        """Create input tensors with rank-specific values."""
        inputs = []
        options = {"dtype": dtype, "device": self.device}

        for i in range(self.num_ranks):
            # Each tensor has rank-specific values
            if dtype == torch.float or dtype == torch.bfloat16:
                tensor = torch.ones(count, **options) * float(i + 1)
            elif dtype == torch.int:
                tensor = torch.ones(count, **options) * int(i + 1)
            elif dtype == torch.int8:
                tensor = torch.ones(count, **options) * int(i + 1)
            inputs.append(tensor)

        return inputs

    def _create_output_tensor(self, count, dtype):
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count, **options)

    def _verify_results(self, output):
        """Verify the results of the scatter operation."""
        # Extract dtype from the tensor
        dtype = output.dtype
        count = output.numel()

        # Expected value for this tensor
        if dtype == torch.float:
            expected = torch.ones(count, dtype=dtype) * float(self.rank + 1)
            self.assertTrue(
                torch.allclose(output.cpu(), expected),
                f"Tensors not close enough for rank {self.rank} tensor",
            )
        else:
            expected = torch.ones(count, dtype=dtype) * int(self.rank + 1)
            self.assertTrue(
                torch.equal(output.cpu(), expected),
                f"Tensors not equal for rank {self.rank} tensor",
            )

    def test_sync_scatter(self):
        """Test synchronous scatter with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_scatter(count, dtype)

    def test_sync_scatter_no_work(self):
        """Test synchronous scatter without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_scatter_no_work(count, dtype)

    def test_async_scatter(self):
        """Test asynchronous scatter with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_scatter(count, dtype)

    def test_async_scatter_early_reset(self):
        """Test asynchronous scatter with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_scatter_early_reset(count, dtype)

    def test_scatter_input_deleted(self):
        """Test asynchronous scatter with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._scatter_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
