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


class AllToAllSingleTest(unittest.TestCase):
    """Test class for all_to_all_single operations in TorchComm."""

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

    def _sync_all_to_all_single(self, count, dtype):
        """Test synchronous all_to_all_single with work object."""
        print(
            f"Testing sync all_to_all_single with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call all_to_all_single
        work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
        work.wait()

        # Verify the results
        self._verify_results(output_tensor)

    def _sync_all_to_all_single_no_work(self, count, dtype):
        """Test synchronous all_to_all_single without work object."""
        print(
            f"Testing sync all_to_all_single without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call all_to_all_single without keeping the work object
        self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)

        # Verify the results
        self._verify_results(output_tensor)

    def _async_all_to_all_single(self, count, dtype):
        """Test asynchronous all_to_all_single with wait."""
        print(
            f"Testing async all_to_all_single with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call all_to_all_single
        work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, True)

        # Wait for the all_to_all_single to complete
        work.wait()

        # Verify the results
        self._verify_results(output_tensor)

    def _async_all_to_all_single_early_reset(self, count, dtype):
        """Test asynchronous all_to_all_single with early reset."""
        print(
            f"Testing async all_to_all_single with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        # Call all_to_all_single
        work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output_tensor)

    def _all_to_all_single_input_deleted(self, count, dtype):
        """Test asynchronous all_to_all_single with input deleted after enqueue."""
        print(
            f"Testing async all_to_all_single with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create output tensor that persists throughout the test
        output_tensor = self._create_output_tensor(count, dtype)

        # Create input tensor and enqueue operation
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_to_all_single with async_op = False
        self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)

        # Delete the input tensor to simulate it going out of scope
        del input_tensor

        # Verify the results
        self._verify_results(output_tensor)

    def _graph_all_to_all_single(self, count, dtype):
        """Test CUDA Graph all_to_all_single."""
        print(
            f"Testing CUDA Graph all_to_all_single with count={count} and dtype={get_dtype_name(dtype)}"
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

            # Capture the reset + all_to_all_single operations in the graph
            with torch.cuda.graph(graph):
                # Call all_to_all_single without keeping the work object
                self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output tensor before graph replay
                output_tensor.copy_(original_output_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensor)

    def _graph_all_to_all_single_input_deleted(self, count, dtype):
        """Test CUDA Graph all_to_all_single with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph all_to_all_single with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
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

            # Create input tensor for graph capture
            input_tensor = self._create_input_tensor(count, dtype)

            # Capture the reset + all_to_all_single operations in the graph
            with torch.cuda.graph(graph):
                # Call all_to_all_single without keeping the work object
                self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)

            # Delete the input tensor to simulate it going out of scope
            del input_tensor

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                # Reset output tensor before each replay
                output_tensor.copy_(original_output_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensor)

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if dtype == torch.float or dtype == torch.bfloat16:
            return torch.ones(count * self.num_ranks, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count * self.num_ranks, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count * self.num_ranks, **options) * int(self.rank + 1)
        return None

    def _create_output_tensor(self, count, dtype):
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count * self.num_ranks, **options)

    def _verify_results(self, output_tensor):
        """Verify the results of the all_to_all_single operation."""
        # Extract count from the tensor
        count = output_tensor.numel() // self.num_ranks

        for i in range(self.num_ranks):
            # For each rank's section in the output tensor
            section = output_tensor[i * count : (i + 1) * count]

            # Expected value for this section
            if section.dtype == torch.float:
                expected = torch.ones(count, dtype=section.dtype) * float(i + 1)
                self.assertTrue(
                    torch.allclose(section.cpu(), expected),
                    f"Tensors not close enough for rank {i} section",
                )
            else:
                expected = torch.ones(count, dtype=section.dtype) * int(i + 1)
                self.assertTrue(
                    torch.equal(section.cpu(), expected),
                    f"Tensors not equal for rank {i} section",
                )

    def test_all_tests(self):
        """Run all tests with all parameter combinations."""

        # CUDA-graph-aware collective coverage was specific to a backend
        # that is no longer built; disabled here.
        runCudaGraphTests = False

        # Nested loops for all parameter combinations
        for count, dtype in itertools.product(self.counts, self.dtypes):
            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}"
            print(f"Running tests with parameters: {test_name}")

            # Run all test functions with clear tracing
            print("Running _sync_all_to_all_single")
            self._sync_all_to_all_single(count, dtype)

            print("Running _sync_all_to_all_single_no_work")
            self._sync_all_to_all_single_no_work(count, dtype)

            print("Running _async_all_to_all_single")
            self._async_all_to_all_single(count, dtype)

            print("Running _async_all_to_all_single_early_reset")
            self._async_all_to_all_single_early_reset(count, dtype)

            print("Running _all_to_all_single_input_deleted")
            self._all_to_all_single_input_deleted(count, dtype)

            if runCudaGraphTests:
                print("Running _graph_all_to_all_single")
                self._graph_all_to_all_single(count, dtype)

                print("Running _graph_all_to_all_single_input_deleted")
                self._graph_all_to_all_single_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
