#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest
from enum import Enum

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    is_full_sweep,
    TorchCommTestWrapper,
)

import torch


class AllToAllvSingleTest(unittest.TestCase):
    """Test class for all_to_all_v_single operations in TorchComm."""

    # Class variables for test parameters
    counts = [4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = (
        [torch.float, torch.bfloat16, torch.int, torch.int8]
        if is_full_sweep()
        else [torch.float, torch.bfloat16]
    )
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

    def _sync_all_to_all_v_single(self, input_split_sizes, output_split_sizes, dtype):
        """Test synchronous all_to_all_v_single with work object."""
        print(f"Testing sync all_to_all_v_single with dtype={get_dtype_name(dtype)}")

        # Create input and output tensors
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)

        # Call all_to_all_v_single
        work = self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, False
        )
        work.wait()

        # Verify the results
        self._verify_results(output_tensor, output_split_sizes)

    def _sync_all_to_all_v_single_no_work(
        self, input_split_sizes, output_split_sizes, dtype
    ):
        """Test synchronous all_to_all_v_single without work object."""
        print(
            f"Testing sync all_to_all_v_single without work object with dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)

        # Call all_to_all_v_single without keeping the work object
        self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, False
        )

        # Verify the results
        self._verify_results(output_tensor, output_split_sizes)

    def _async_all_to_all_v_single(self, input_split_sizes, output_split_sizes, dtype):
        """Test asynchronous all_to_all_v_single with wait."""
        print(f"Testing async all_to_all_v_single with dtype={get_dtype_name(dtype)}")

        # Create input and output tensors
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)

        # Call all_to_all_v_single
        work = self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, True
        )

        # Wait for the all_to_all_v_single to complete
        work.wait()

        # Verify the results
        self._verify_results(output_tensor, output_split_sizes)

    def _async_all_to_all_v_single_early_reset(
        self, input_split_sizes, output_split_sizes, dtype
    ):
        """Test asynchronous all_to_all_v_single with early reset."""
        print(
            f"Testing async all_to_all_v_single with early reset with dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)

        # Call all_to_all_v_single
        work = self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, True
        )

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output_tensor, output_split_sizes)

    def _all_to_all_v_single_input_deleted(
        self, input_split_sizes, output_split_sizes, dtype
    ):
        """Test asynchronous all_to_all_v_single with input deleted after enqueue."""
        print(
            f"Testing async all_to_all_v_single with input deleted after enqueue with dtype={get_dtype_name(dtype)}"
        )

        # Create output tensor that persists throughout the test
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)

        # Create input tensor and enqueue operation
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)

        # Call all_to_all_v_single with async_op = False
        self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, False
        )

        # Delete the input tensor to simulate it going out of scope
        del input_tensor

        # Verify the results
        self._verify_results(output_tensor, output_split_sizes)

    def _graph_all_to_all_v_single(self, input_split_sizes, output_split_sizes, dtype):
        """Test CUDA Graph all_to_all_v_single."""
        print(
            f"Testing CUDA Graph all_to_all_v_single with dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input and output tensors AFTER setting non-default stream but BEFORE graph capture
            input_tensor = self._create_input_tensor(input_split_sizes, dtype)
            output_tensor = self._create_output_tensor(output_split_sizes, dtype)

            # Store original values to reset with
            original_output_tensor = output_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the reset + all_to_all_v_single operations in the graph
            with torch.cuda.graph(graph):
                # Reset tensor to original values inside the graph
                output_tensor.copy_(original_output_tensor)

                # Call all_to_all_v_single without keeping the work object
                self.torchcomm.all_to_all_v_single(
                    output_tensor,
                    input_tensor,
                    output_split_sizes,
                    input_split_sizes,
                    False,
                )

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output tensor before graph replay
                output_tensor.copy_(original_output_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensor, output_split_sizes)

    def _graph_all_to_all_v_single_input_deleted(
        self, input_split_sizes, output_split_sizes, dtype
    ):
        """Test CUDA Graph all_to_all_v_single with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph all_to_all_v_single with input deleted after graph creation with dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create output tensor that persists throughout the test
            output_tensor = self._create_output_tensor(output_split_sizes, dtype)

            # Store original values to reset with
            original_output_tensor = output_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensor for graph capture
            input_tensor = self._create_input_tensor(input_split_sizes, dtype)

            # Capture the reset + all_to_all_v_single operations in the graph
            with torch.cuda.graph(graph):
                # Reset tensor to original values inside the graph
                output_tensor.copy_(original_output_tensor)

                # Call all_to_all_v_single without keeping the work object
                self.torchcomm.all_to_all_v_single(
                    output_tensor,
                    input_tensor,
                    output_split_sizes,
                    input_split_sizes,
                    False,
                )

            # Delete the input tensor to simulate it going out of scope
            del input_tensor

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                # Reset output tensor before each replay
                output_tensor.copy_(original_output_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensor, output_split_sizes)

    def _sync_all_to_all_v_single_multi_dim_tensor(
        self, input_split_sizes_, output_split_sizes_, dtype
    ):
        """Test synchronous all_to_all_v_single with multi-dimensional tensor."""
        print(
            f"Testing sync all_to_all_v_single with multi-dim tensor with dtype={get_dtype_name(dtype)}"
        )

        input_split_sizes = input_split_sizes_.copy()
        output_split_sizes = output_split_sizes_.copy()

        # Create input and output tensors
        input_tensor = self._create_input_tensor(input_split_sizes, dtype)
        input_tensor = input_tensor.reshape(input_tensor.numel() // 2, 2)
        output_tensor = self._create_output_tensor(output_split_sizes, dtype)
        output_tensor = output_tensor.reshape(output_tensor.numel() // 2, 2)

        # Reduce each value in input_split_sizes and output_split_sizes by half
        for i in range(len(input_split_sizes)):
            if input_split_sizes[i] % 2 != 0:
                print("Input size must be divisible by 2 for multi-dim tensor test")
                return
            else:
                input_split_sizes[i] //= 2

        for i in range(len(output_split_sizes)):
            if output_split_sizes[i] % 2 != 0:
                print("Output size must be divisible by 2 for multi-dim tensor test")
                return
            else:
                output_split_sizes[i] //= 2

        # Call all_to_all_v_single
        work = self.torchcomm.all_to_all_v_single(
            output_tensor, input_tensor, output_split_sizes, input_split_sizes, False
        )
        work.wait()

        output_tensor = output_tensor.reshape(output_tensor.numel())

        # Verify the results with the original output_split_sizes
        self._verify_results(output_tensor, output_split_sizes_)

    def _create_input_tensor(self, input_split_sizes, dtype):
        """Create input tensor with rank-specific values."""
        total_size = sum(input_split_sizes)
        options = {"dtype": dtype, "device": self.device}
        input_tensor = torch.zeros(total_size, **options)

        # Fill each section with rank-specific values
        offset = 0
        for i in range(self.num_ranks):
            if input_split_sizes[i] > 0:
                section = input_tensor[offset : offset + input_split_sizes[i]]
                if dtype == torch.float or dtype == torch.bfloat16:
                    section.fill_(float(self.rank * 100 + i + 1))
                elif dtype == torch.int:
                    section.fill_(int(self.rank * 100 + i + 1))
                elif dtype == torch.int8:
                    section.fill_(int((self.rank * 10 + i + 1) % 128))
            offset += input_split_sizes[i]

        return input_tensor

    def _create_output_tensor(self, output_split_sizes, dtype):
        """Create output tensor to store results.

        Pre-fill with a sentinel (NaN for float, -1 for ints) so unwritten
        positions are caught by verification — using torch.zeros would hide
        an off-by-one displacement bug where the buffer happens to already
        contain 0.
        """
        total_size = sum(output_split_sizes)
        options = {"dtype": dtype, "device": self.device}
        output = torch.empty(total_size, **options)
        if dtype in (torch.float, torch.double, torch.half, torch.bfloat16):
            output.fill_(float("nan"))
        else:
            output.fill_(-1)
        return output

    def _verify_results(self, output_tensor, output_split_sizes):
        """Verify the results of the all_to_all_v_single operation."""
        offset = 0
        for i in range(self.num_ranks):
            if output_split_sizes[i] > 0:
                # For each rank's section in the output tensor
                section = output_tensor[offset : offset + output_split_sizes[i]]

                # Expected value: what rank i would have sent to this rank
                if output_tensor.dtype == torch.float:
                    expected_value = float(i * 100 + self.rank + 1)
                    expected = torch.full(
                        (output_split_sizes[i],), expected_value, dtype=section.dtype
                    )
                    self.assertTrue(
                        torch.allclose(section.cpu(), expected),
                        f"Tensors not close enough for rank {i} section",
                    )
                elif output_tensor.dtype == torch.bfloat16:
                    # Each section is filled with a single constant value, so the
                    # bf16 representation of `expected_value` should bit-match the
                    # received data. NaN-init means a missed write surfaces as
                    # NaN, which torch.equal rejects.
                    expected_value = float(i * 100 + self.rank + 1)
                    expected = torch.full(
                        (output_split_sizes[i],), expected_value, dtype=section.dtype
                    )
                    self.assertTrue(
                        torch.equal(section.cpu(), expected),
                        f"bfloat16 tensors not equal for rank {i} section",
                    )
                elif output_tensor.dtype == torch.int:
                    expected_value = int(i * 100 + self.rank + 1)
                    expected = torch.full(
                        (output_split_sizes[i],), expected_value, dtype=section.dtype
                    )
                    self.assertTrue(
                        torch.equal(section.cpu(), expected),
                        f"Tensors not equal for rank {i} section",
                    )
                elif output_tensor.dtype == torch.int8:
                    expected_value = int((i * 10 + self.rank + 1) % 128)
                    expected = torch.full(
                        (output_split_sizes[i],), expected_value, dtype=section.dtype
                    )
                    self.assertTrue(
                        torch.equal(section.cpu(), expected),
                        f"Tensors not equal for rank {i} section",
                    )
            offset += output_split_sizes[i]

    def _create_uniform_sizes(self, count):
        """Create size vectors for uniform pattern."""
        # Test with uniform sizes
        input_sizes = [count] * self.num_ranks
        output_sizes = [count] * self.num_ranks
        return input_sizes, output_sizes

    def _create_variable_sizes(self, count):
        """Create size vectors for variable pattern.

        Each (src, dst) pair gets a distinct size so the test exercises:
          - different total tensor sizes per rank
          - different per-destination split sizes within a single rank
          - different per-source split sizes within a single rank

        Rank i sends (i+1)*(j+1)*count elements to rank j, so rank j
        expects to receive (i+1)*(j+1)*count elements from rank i.
        """
        input_sizes = []
        output_sizes = []
        for j in range(self.num_ranks):
            # This rank (src=self.rank) sends to rank j.
            input_sizes.append((self.rank + 1) * (j + 1) * count)
        for i in range(self.num_ranks):
            # This rank (dst=self.rank) receives from rank i.
            output_sizes.append((i + 1) * (self.rank + 1) * count)
        return input_sizes, output_sizes

    def _create_zero_sizes(self, count):
        """Create size vectors for zero sizes pattern."""
        # Test with some zero sizes - ensure symmetric pattern
        input_sizes = []
        output_sizes = []
        for i in range(self.num_ranks):
            # Create a pattern where some communications have zero size
            # If this rank sends 0 to rank i, then rank i sends 0 to this rank
            size = 0 if (self.rank + i) % 3 == 0 else count
            input_sizes.append(size)

            # This rank receives from rank i what rank i sends to this rank
            recv_size = 0 if (i + self.rank) % 3 == 0 else count
            output_sizes.append(recv_size)
        return input_sizes, output_sizes

    def _create_asymmetric_sizes(self, count):
        """Create size vectors for asymmetric pattern."""
        # Create an asymmetric pattern where even ranks send data but odd ranks don't
        # However, odd ranks receive data from even ranks
        input_sizes = []
        output_sizes = []

        for i in range(self.num_ranks):
            if self.rank % 2 == 0:
                # Even ranks send data to all ranks
                input_sizes.append(count)
            else:
                # Odd ranks don't send any data
                input_sizes.append(0)

            if i % 2 == 0:
                # All ranks receive data from even ranks
                output_sizes.append(count)
            else:
                # All ranks don't receive from odd ranks (since they don't send)
                output_sizes.append(0)
        return input_sizes, output_sizes

    def test_all_tests(self):
        """Run all tests with all parameter combinations."""

        # Define enum for size patterns
        class SizePattern(Enum):
            UNIFORM = "Uniform"
            VARIABLE = "Variable"
            ZERO_SIZES = "ZeroSizes"

        # Define size patterns to test (excluding AllZero which is handled separately)
        size_pattern_creators = {
            SizePattern.UNIFORM: self._create_uniform_sizes,
            SizePattern.VARIABLE: self._create_variable_sizes,
            SizePattern.ZERO_SIZES: self._create_zero_sizes,
        }

        # CUDA-graph-aware collective coverage was specific to a backend
        # that is no longer built; disabled here.
        runCudaGraphTests = False

        # Nested loops for all parameter combinations (counts x patterns x dtypes)
        for count, pattern, dtype in itertools.product(
            self.counts, size_pattern_creators.keys(), self.dtypes
        ):
            # Create size vectors based on pattern and count
            input_sizes, output_sizes = size_pattern_creators[pattern](count)

            # Create a descriptive test name for better test output
            test_name = f"{pattern.value}_{count}_{get_dtype_name(dtype)}"
            print(f"Running tests with parameters: {test_name}")

            # Run all test functions with clear tracing
            print("Running _sync_all_to_all_v_single")
            self._sync_all_to_all_v_single(input_sizes, output_sizes, dtype)

            print("Running _sync_all_to_all_v_single_no_work")
            self._sync_all_to_all_v_single_no_work(input_sizes, output_sizes, dtype)

            print("Running _async_all_to_all_v_single")
            self._async_all_to_all_v_single(input_sizes, output_sizes, dtype)

            print("Running _async_all_to_all_v_single_early_reset")
            self._async_all_to_all_v_single_early_reset(
                input_sizes, output_sizes, dtype
            )

            # Only run multi-dim tensor test for uniform sizes where all values are divisible by 2
            if pattern == SizePattern.UNIFORM and count % 2 == 0:
                print("Running _sync_all_to_all_v_single_multi_dim_tensor")
                self._sync_all_to_all_v_single_multi_dim_tensor(
                    input_sizes, output_sizes, dtype
                )

            if runCudaGraphTests:
                print("Running _graph_all_to_all_v_single")
                self._graph_all_to_all_v_single(input_sizes, output_sizes, dtype)

        # Handle AllZero separately as requested
        for dtype in self.dtypes:
            # Test with all zero sizes
            input_sizes = [0] * self.num_ranks
            output_sizes = [0] * self.num_ranks

            # Create a descriptive test name for better test output
            test_name = f"AllZero_{get_dtype_name(dtype)}"
            print(f"Running tests with parameters: {test_name}")

            # Run all test functions with clear tracing
            print("Running _sync_all_to_all_v_single")
            self._sync_all_to_all_v_single(input_sizes, output_sizes, dtype)

            print("Running _sync_all_to_all_v_single_no_work")
            self._sync_all_to_all_v_single_no_work(input_sizes, output_sizes, dtype)

            print("Running _async_all_to_all_v_single")
            self._async_all_to_all_v_single(input_sizes, output_sizes, dtype)

            print("Running _async_all_to_all_v_single_early_reset")
            self._async_all_to_all_v_single_early_reset(
                input_sizes, output_sizes, dtype
            )

            print("Running _all_to_all_v_single_input_deleted")
            self._all_to_all_v_single_input_deleted(input_sizes, output_sizes, dtype)

            if runCudaGraphTests:
                print("Running _graph_all_to_all_v_single")
                self._graph_all_to_all_v_single(input_sizes, output_sizes, dtype)

                print("Running _graph_all_to_all_v_single_input_deleted")
                self._graph_all_to_all_v_single_input_deleted(
                    input_sizes, output_sizes, dtype
                )

        # Test asymmetric communication: some ranks have all zero inputs but non-zero outputs
        for count in self.counts:
            for dtype in self.dtypes:
                input_sizes, output_sizes = self._create_asymmetric_sizes(count)

                # Create a descriptive test name for better test output
                test_name = f"AsymmetricZeroInput_{count}_{get_dtype_name(dtype)}"
                print(f"Running tests with parameters: {test_name}")

                # Run all test functions with clear tracing
                print("Running _sync_all_to_all_v_single")
                self._sync_all_to_all_v_single(input_sizes, output_sizes, dtype)

                print("Running _sync_all_to_all_v_single_no_work")
                self._sync_all_to_all_v_single_no_work(input_sizes, output_sizes, dtype)

                print("Running _async_all_to_all_v_single")
                self._async_all_to_all_v_single(input_sizes, output_sizes, dtype)

                print("Running _async_all_to_all_v_single_early_reset")
                self._async_all_to_all_v_single_early_reset(
                    input_sizes, output_sizes, dtype
                )

                print("Running _all_to_all_v_single_input_deleted")
                self._all_to_all_v_single_input_deleted(
                    input_sizes, output_sizes, dtype
                )

                if runCudaGraphTests:
                    print("Running _graph_all_to_all_v_single")
                    self._graph_all_to_all_v_single(input_sizes, output_sizes, dtype)

                    print("Running _graph_all_to_all_v_single_input_deleted")
                    self._graph_all_to_all_v_single_input_deleted(
                        input_sizes, output_sizes, dtype
                    )


if __name__ == "__main__":
    unittest.main()
