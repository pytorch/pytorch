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


class BroadcastTest(unittest.TestCase):
    """Test class for broadcast operations in TorchComm."""

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

    def _sync_broadcast(self, count, dtype):
        """Test synchronous broadcast with work object."""
        print(
            f"Testing sync broadcast with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 99

        # Create tensor with different values based on rank
        tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

        # Call broadcast
        work = self.torchcomm.broadcast(tensor, root_rank, False)
        work.wait()

        # Verify the results
        self._verify_broadcast_results(tensor, root_value)

    def _sync_broadcast_no_work(self, count, dtype):
        """Test synchronous broadcast without work object."""
        print(
            f"Testing sync broadcast without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 99

        # Create tensor with different values based on rank
        tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

        # Call broadcast without keeping the work object
        self.torchcomm.broadcast(tensor, root_rank, False)

        # Verify the results
        self._verify_broadcast_results(tensor, root_value)

    def _async_broadcast(self, count, dtype):
        """Test asynchronous broadcast with wait."""
        print(
            f"Testing async broadcast with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 42

        # Create tensor with different values based on rank
        tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

        # Call broadcast
        work = self.torchcomm.broadcast(tensor, root_rank, True)

        # Wait for the broadcast to complete
        work.wait()

        # Verify the results
        self._verify_broadcast_results(tensor, root_value)

    def _async_broadcast_early_reset(self, count, dtype):
        """Test asynchronous broadcast with early reset."""
        print(
            f"Testing async broadcast with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 42

        # Create tensor with different values based on rank
        tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

        # Call broadcast
        work = self.torchcomm.broadcast(tensor, root_rank, True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_broadcast_results(tensor, root_value)

    def _broadcast_input_deleted(self, count, dtype):
        """Test asynchronous broadcast with input deleted after enqueue."""
        print(
            f"Testing async broadcast with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 42

        # Create tensor and enqueue operation
        tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

        # Call broadcast with async_op = False
        self.torchcomm.broadcast(tensor, root_rank, False)

        # Delete the tensor to simulate it going out of scope
        del tensor

        # Note: For broadcast, the operation is in-place, so we need to create a new tensor
        # to verify results since the original was deleted. This test primarily validates
        # that the operation completes without crashing when input is deleted.

    def _graph_broadcast(self, count, dtype):
        """Test CUDA Graph broadcast."""
        print(
            f"Testing CUDA Graph broadcast with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            root_rank = 0
            root_value = 99

            # Create tensor with different values based on rank AFTER setting non-default stream but BEFORE graph capture
            tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)
            # For non-root ranks, create original tensor to reset to
            if self.rank != root_rank:
                original_tensor = tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the broadcast operation in the graph
            with torch.cuda.graph(graph):
                # Call broadcast without keeping the work object
                self.torchcomm.broadcast(tensor, root_rank, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset tensor before each replay (non-root ranks need fresh data)
                if self.rank != root_rank:
                    tensor.copy_(original_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_broadcast_results(tensor, root_value)

    def _graph_broadcast_input_deleted(self, count, dtype):
        """Test CUDA Graph broadcast with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph broadcast with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        root_value = 99

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create tensor in a limited scope
            tensor = self._create_broadcast_tensor(root_rank, root_value, count, dtype)

            # Capture the broadcast operation in the graph
            with torch.cuda.graph(graph):
                # Call broadcast without keeping the work object
                self.torchcomm.broadcast(tensor, root_rank, False)

            # Tensor goes out of scope here and gets deleted
            del tensor

        # Replay the captured graph multiple times even though tensor is deleted
        for _ in range(self.num_replays):
            graph.replay()

            # Note: Cannot verify results since tensor is deleted
            # This test validates that the graph replay completes without crashing

    def _create_broadcast_tensor(self, root_rank, value, count, dtype):
        """Create tensor for broadcast with different values based on rank."""
        options = {"dtype": dtype, "device": self.device}

        # Initialize tensor based on dtype
        if dtype == torch.float or dtype == torch.bfloat16:
            if self.rank == root_rank:
                return torch.ones(count, **options) * float(value)
            else:
                return torch.zeros(count, **options)
        elif dtype == torch.int:
            if self.rank == root_rank:
                return torch.ones(count, **options) * int(value)
            else:
                return torch.zeros(count, **options)
        elif dtype == torch.int8:
            if self.rank == root_rank:
                return torch.ones(count, **options) * int(value)
            else:
                return torch.zeros(count, **options)
        return None

    def _verify_broadcast_results(self, tensor, value):
        """Verify the results of the broadcast operation."""
        # Extract dtype from the tensor
        dtype = tensor.dtype
        count = tensor.numel()

        # Expected value for this tensor
        if dtype == torch.float:
            expected = torch.ones(count, dtype=dtype) * float(value)
            self.assertTrue(
                torch.allclose(tensor.cpu(), expected),
                f"Tensors not close enough for broadcast with value {value}",
            )
        else:
            expected = torch.ones(count, dtype=dtype) * int(value)
            self.assertTrue(
                torch.equal(tensor.cpu(), expected),
                f"Tensors not equal for broadcast with value {value}",
            )

    def test_sync_broadcast(self):
        """Test synchronous broadcast with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_broadcast(count, dtype)

    def test_sync_broadcast_no_work(self):
        """Test synchronous broadcast without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_broadcast_no_work(count, dtype)

    def test_async_broadcast(self):
        """Test asynchronous broadcast with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_broadcast(count, dtype)

    def test_async_broadcast_early_reset(self):
        """Test asynchronous broadcast with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_broadcast_early_reset(count, dtype)

    def test_broadcast_input_deleted(self):
        """Test asynchronous broadcast with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._broadcast_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
