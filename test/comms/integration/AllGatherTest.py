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


class AllGatherTest(unittest.TestCase):
    """Test class for all_gather operations in TorchComm."""

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

    def _sync_all_gather(self, count, dtype):
        """Test synchronous all_gather with work object."""
        print(
            f"Testing sync all_gather with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)

        # Call all_gather
        work = self.torchcomm.all_gather(output_tensors, input_tensor, False)
        work.wait()

        # Verify the results
        self._verify_results(output_tensors)

    def _sync_all_gather_no_work(self, count, dtype):
        """Test synchronous all_gather without work object."""
        print(
            f"Testing sync all_gather without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)

        # Call all_gather without keeping the work object
        self.torchcomm.all_gather(output_tensors, input_tensor, False)

        # Verify the results
        self._verify_results(output_tensors)

    def _async_all_gather(self, count, dtype):
        """Test asynchronous all_gather with wait."""
        print(
            f"Testing async all_gather with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)

        # Call all_gather
        work = self.torchcomm.all_gather(output_tensors, input_tensor, True)

        # Wait for the all_gather to complete
        work.wait()

        # Verify the results
        self._verify_results(output_tensors)

    def _async_all_gather_early_reset(self, count, dtype):
        """Test asynchronous all_gather with early reset."""
        print(
            f"Testing async all_gather with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensors = self._create_output_tensors(count, dtype)

        # Call all_gather
        work = self.torchcomm.all_gather(output_tensors, input_tensor, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(output_tensors)

    def _all_gather_input_deleted(self, count, dtype):
        """Test asynchronous all_gather with input deleted after enqueue."""
        print(
            f"Testing async all_gather with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create output tensors that persist throughout the test
        output_tensors = self._create_output_tensors(count, dtype)

        # Create input tensor and enqueue operation
        input_tensor = self._create_input_tensor(count, dtype)

        # Call all_gather with async_op = False
        self.torchcomm.all_gather(output_tensors, input_tensor, False)

        # Delete the input tensor to simulate it going out of scope
        del input_tensor

        # Verify the results
        self._verify_results(output_tensors)

    def _graph_all_gather(self, count, dtype):
        """Test CUDA Graph all_gather."""
        print(
            f"Testing CUDA Graph all_gather with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create input and output tensors AFTER setting non-default stream but BEFORE graph capture
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensors = self._create_output_tensors(count, dtype)
            original_output_tensors = [tensor.clone() for tensor in output_tensors]

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the all_gather operation in the graph
            with torch.cuda.graph(graph):
                # Call all_gather without keeping the work object
                self.torchcomm.all_gather(output_tensors, input_tensor, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output tensors before each replay
                for i, output_tensor in enumerate(output_tensors):
                    output_tensor.copy_(original_output_tensors[i])

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensors)

    def _graph_all_gather_input_deleted(self, count, dtype):
        """Test CUDA Graph all_gather with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph all_gather with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )
        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create output tensors that persist throughout the test
            output_tensors = self._create_output_tensors(count, dtype)
            original_output_tensors = [tensor.clone() for tensor in output_tensors]

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create input tensor for graph capture
            input_tensor = self._create_input_tensor(count, dtype)

            # Capture the all_gather operation in the graph
            with torch.cuda.graph(graph):
                # Call all_gather without keeping the work object
                self.torchcomm.all_gather(output_tensors, input_tensor, False)

            # Delete the input tensor to simulate it going out of scope
            del input_tensor

            # Replay the captured graph multiple times even though input is deleted
            for _ in range(self.num_replays):
                # Reset output tensors before each replay
                for i, output_tensor in enumerate(output_tensors):
                    output_tensor.copy_(original_output_tensors[i])

                graph.replay()

                # Verify the results after each replay
                self._verify_results(output_tensors)

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

    def _create_output_tensors(self, count, dtype):
        """Create output tensors to store results."""
        options = {"dtype": dtype, "device": self.device}
        output_tensors = []
        for _ in range(self.num_ranks):
            output_tensors.append(torch.zeros(count, **options))
        return output_tensors

    def _verify_results(self, output_tensors):
        """Verify the results of the all_gather operation."""
        for i in range(self.num_ranks):
            # Extract dtype from the tensor
            dtype = output_tensors[i].dtype
            count = output_tensors[i].numel()

            # Expected value for this tensor
            if dtype == torch.float:
                expected = torch.ones(count, dtype=dtype) * float(i + 1)
                self.assertTrue(
                    torch.allclose(output_tensors[i].cpu(), expected),
                    f"Tensors not close enough for rank {i} tensor",
                )
            else:
                expected = torch.ones(count, dtype=dtype) * int(i + 1)
                self.assertTrue(
                    torch.equal(output_tensors[i].cpu(), expected),
                    f"Tensors not equal for rank {i} tensor",
                )

    def test_sync_all_gather(self):
        """Test synchronous all_gather with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_all_gather(count, dtype)

    def test_sync_all_gather_no_work(self):
        """Test synchronous all_gather without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_all_gather_no_work(count, dtype)

    def test_async_all_gather(self):
        """Test asynchronous all_gather with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_all_gather(count, dtype)

    def test_async_all_gather_early_reset(self):
        """Test asynchronous all_gather with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_all_gather_early_reset(count, dtype)

    def test_all_gather_input_deleted(self):
        """Test asynchronous all_gather with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._all_gather_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
