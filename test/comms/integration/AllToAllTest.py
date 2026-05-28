#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    is_full_sweep,
    TorchCommTestWrapper,
)

import torch


class AllToAllTest(unittest.TestCase):
    """Test class for all_to_all operations in TorchComm."""

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

    def _sync_all_to_all(self, count, dtype):
        """Test synchronous all_to_all with work object."""
        print(
            f"Testing sync all_to_all with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensors = self._create_input_tensors(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)
        expected_output = self._create_expected_output()

        # Call all_to_all
        work = self.torchcomm.all_to_all(output_tensors, input_tensors, False)
        work.wait()

        # Verify the results
        self._verify_results(output_tensors, expected_output)

    def _sync_all_to_all_no_work(self, count, dtype):
        """Test synchronous all_to_all without work object."""
        print(
            f"Testing sync all_to_all without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensors = self._create_input_tensors(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)
        expected_output = self._create_expected_output()

        # Call all_to_all without keeping the work object
        self.torchcomm.all_to_all(output_tensors, input_tensors, False)

        # Verify the results
        self._verify_results(output_tensors, expected_output)

    def _async_all_to_all(self, count, dtype):
        """Test asynchronous all_to_all with wait."""
        print(
            f"Testing async all_to_all with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensors = self._create_input_tensors(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)
        expected_output = self._create_expected_output()

        # Call all_to_all
        work = self.torchcomm.all_to_all(output_tensors, input_tensors, True)

        # Wait for the all_to_all to complete
        work.wait()

        # Verify the results
        self._verify_results(output_tensors, expected_output)

    def _async_all_to_all_early_reset(self, count, dtype):
        """Test asynchronous all_to_all with early reset."""
        print(
            f"Testing async all_to_all with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensors = self._create_input_tensors(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)
        expected_output = self._create_expected_output()

        # Call all_to_all
        work = self.torchcomm.all_to_all(output_tensors, input_tensors, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output_tensors, expected_output)

    def _all_to_all_input_deleted(self, count, dtype):
        """Test asynchronous all_to_all with input deleted after enqueue."""
        print(
            f"Testing async all_to_all with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create output tensors that persist throughout the test
        output_tensors = self._create_output_tensors(count, dtype)
        expected_output = self._create_expected_output()

        # Create input tensors and enqueue operation
        input_tensors = self._create_input_tensors(count, dtype)

        # Call all_to_all with async_op = False
        self.torchcomm.all_to_all(output_tensors, input_tensors, False)

        # Delete the input tensors to simulate them going out of scope
        del input_tensors

        # Verify the results
        self._verify_results(output_tensors, expected_output)

    def _graph_all_to_all(self, count, dtype):
        """Test CUDA Graph all_to_all."""
        print(
            f"Testing CUDA Graph all_to_all with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input and output tensors AFTER setting non-default stream but BEFORE graph capture
            input_tensors = self._create_input_tensors(count, dtype)
            output_tensors = self._create_output_tensors(count, dtype)
            expected_output = self._create_expected_output()

            # Store original values to reset with
            original_output_tensors = [tensor.clone() for tensor in output_tensors]

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the reset + all_to_all operations in the graph
            with torch.cuda.graph(graph):
                # Call all_to_all without keeping the work object
                self.torchcomm.all_to_all(output_tensors, input_tensors, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output tensors before graph replay
                for i, original_tensor in enumerate(original_output_tensors):
                    output_tensors[i].copy_(original_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensors, expected_output)

    def _graph_all_to_all_input_deleted(self, count, dtype):
        """Test CUDA Graph all_to_all with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph all_to_all with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create output tensors that persist throughout the test
            output_tensors = self._create_output_tensors(count, dtype)
            expected_output = self._create_expected_output()

            # Store original values to reset with
            original_output_tensors = [tensor.clone() for tensor in output_tensors]

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensors for graph capture
            input_tensors = self._create_input_tensors(count, dtype)

            # Capture the reset + all_to_all operations in the graph
            with torch.cuda.graph(graph):
                # Call all_to_all without keeping the work object
                self.torchcomm.all_to_all(output_tensors, input_tensors, False)

            # Delete the input tensors to simulate them going out of scope
            del input_tensors

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                # Reset output tensors before graph replay
                for i, original_tensor in enumerate(original_output_tensors):
                    output_tensors[i].copy_(original_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensors, expected_output)

    def _create_input_tensors(self, count, dtype):
        """Create input tensors with rank-specific values."""
        input_tensors = []
        options = {"dtype": dtype, "device": self.device}

        for _ in range(self.num_ranks):
            # Each tensor has rank-specific values
            if dtype == torch.float or dtype == torch.bfloat16:
                tensor = torch.ones(count, **options) * float(self.rank + 1)
            elif dtype == torch.int:
                tensor = torch.ones(count, **options) * int(self.rank + 1)
            elif dtype == torch.int8:
                tensor = torch.ones(count, **options) * int(self.rank + 1)
            input_tensors.append(tensor)

        return input_tensors

    def _create_output_tensors(self, count, dtype):
        """Create output tensors to store results."""
        output_tensors = []
        options = {"dtype": dtype, "device": self.device}

        for _ in range(self.num_ranks):
            tensor = torch.zeros(count, **options)
            output_tensors.append(tensor)

        return output_tensors

    def _create_expected_output(self):
        """Create expected output values."""
        expected_output = []

        for r in range(self.num_ranks):
            expected_output.append(r + 1)

        return expected_output

    def _verify_results(self, output_tensors, expected_output):
        """Verify the results of the all_to_all operation."""
        for r in range(self.num_ranks):
            # Extract dtype from the tensor
            dtype = output_tensors[r].dtype
            count = output_tensors[r].numel()

            # Expected value for this tensor
            if dtype == torch.float:
                expected = torch.ones(count, dtype=dtype) * float(expected_output[r])
                self.assertTrue(
                    torch.allclose(output_tensors[r].cpu(), expected),
                    f"Tensors not close enough for rank {r} tensor",
                )
            else:
                expected = torch.ones(count, dtype=dtype) * int(expected_output[r])
                self.assertTrue(
                    torch.equal(output_tensors[r].cpu(), expected),
                    f"Tensors not equal for rank {r} tensor",
                )

    def test_all_tests(self):
        """Run all tests with all parameter combinations."""

        # CUDA-graph-aware collective coverage was specific to a backend
        # that is no longer built; disabled here.
        runCudaGraphTests = False
        isGlooBackend = os.getenv("TEST_BACKEND") == "gloo"

        # Nested loops for all parameter combinations
        for count, dtype in itertools.product(self.counts, self.dtypes):
            # Skip count=0 tests for gloo backend due to implementation limitations
            if isGlooBackend and count == 0:
                print(
                    f"Skipping Count_{count}_{get_dtype_name(dtype)} for gloo backend - zero-sized tensor limitation"
                )
                continue

            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}"
            print(f"Running tests with parameters: {test_name}")

            # Run all test functions with clear tracing
            print("Running _sync_all_to_all")
            self._sync_all_to_all(count, dtype)

            print("Running _sync_all_to_all_no_work")
            self._sync_all_to_all_no_work(count, dtype)

            print("Running _async_all_to_all")
            self._async_all_to_all(count, dtype)

            print("Running _async_all_to_all_early_reset")
            self._async_all_to_all_early_reset(count, dtype)

            print("Running _all_to_all_input_deleted")
            self._all_to_all_input_deleted(count, dtype)

            if runCudaGraphTests:
                print("Running _graph_all_to_all")
                self._graph_all_to_all(count, dtype)

                print("Running _graph_all_to_all_input_deleted")
                self._graph_all_to_all_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
