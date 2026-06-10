#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)

import torch


class BatchSendRecvTest(unittest.TestCase):
    """Test class for batch SendRecv operations in TorchComm."""

    # Class variables for test parameters
    counts = [4]  # Using smaller counts to avoid potential deadlocks in tests
    dtypes = [torch.float, torch.int, torch.int8]
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

    def _sync_batch_sendrecv(self, count, dtype):
        """Test synchronous batch SendRecv operations with work object."""
        print(
            f"Testing sync batch SendRecv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to next rank and receives from previous rank
        next_rank = (self.rank + 1) % self.num_ranks
        prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create batch operation object
        batch_op = self.torchcomm.batch_op_create()

        # Create tensors for batch operations (2 send and 2 recv operations)
        send_tensors = []
        recv_tensors = []

        for i in range(2):
            # Create send tensor with unique values
            send_tensor = self._create_send_tensor(count, dtype, i)
            send_tensors.append(send_tensor)
            batch_op.send(send_tensor, next_rank)

            # Create recv tensor
            recv_tensor = self._create_recv_tensor(count, dtype)
            recv_tensors.append(recv_tensor)
            batch_op.recv(recv_tensor, prev_rank)

        # Issue batch operations synchronously
        work = batch_op.issue(False)

        work.wait()

        # Verify the results
        self._verify_results(recv_tensors, prev_rank)

    def _sync_batch_sendrecv_no_work(self, count, dtype):
        """Test synchronous batch SendRecv operations without work object."""
        print(
            f"Testing sync batch SendRecv without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to next rank and receives from previous rank
        next_rank = (self.rank + 1) % self.num_ranks
        prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create batch operation object
        batch_op = self.torchcomm.batch_op_create()

        # Create tensors for batch operations (2 send and 2 recv operations)
        send_tensors = []
        recv_tensors = []

        for i in range(2):
            # Create send tensor with unique values
            send_tensor = self._create_send_tensor(count, dtype, i)
            send_tensors.append(send_tensor)
            batch_op.send(send_tensor, next_rank)

            # Create recv tensor
            recv_tensor = self._create_recv_tensor(count, dtype)
            recv_tensors.append(recv_tensor)
            batch_op.recv(recv_tensor, prev_rank)

        # Issue batch operations synchronously without storing work object
        batch_op.issue(False)

        # Verify the results
        self._verify_results(recv_tensors, prev_rank)

    def _async_batch_sendrecv(self, count, dtype):
        """Test asynchronous batch SendRecv operations with wait."""
        print(
            f"Testing async batch SendRecv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to next rank and receives from previous rank
        next_rank = (self.rank + 1) % self.num_ranks
        prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create batch operation object
        batch_op = self.torchcomm.batch_op_create()

        # Create tensors for batch operations (2 send and 2 recv operations)
        send_tensors = []
        recv_tensors = []

        for i in range(2):
            # Create send tensor with unique values
            send_tensor = self._create_send_tensor(count, dtype, i)
            send_tensors.append(send_tensor)
            batch_op.send(send_tensor, next_rank)

            # Create recv tensor
            recv_tensor = self._create_recv_tensor(count, dtype)
            recv_tensors.append(recv_tensor)
            batch_op.recv(recv_tensor, prev_rank)

        # Issue batch operations asynchronously
        work = batch_op.issue(True)

        # Wait for completion
        work.wait()

        # Verify the results
        self._verify_results(recv_tensors, prev_rank)

    def _async_batch_sendrecv_early_reset(self, count, dtype):
        """Test asynchronous batch SendRecv operations with early reset."""
        print(
            f"Testing async batch SendRecv with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to next rank and receives from previous rank
        next_rank = (self.rank + 1) % self.num_ranks
        prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create batch operation object
        batch_op = self.torchcomm.batch_op_create()

        # Create tensors for batch operations (2 send and 2 recv operations)
        send_tensors = []
        recv_tensors = []

        for i in range(2):
            # Create send tensor with unique values
            send_tensor = self._create_send_tensor(count, dtype, i)
            send_tensors.append(send_tensor)
            batch_op.send(send_tensor, next_rank)

            # Create recv tensor
            recv_tensor = self._create_recv_tensor(count, dtype)
            recv_tensors.append(recv_tensor)
            batch_op.recv(recv_tensor, prev_rank)

        # Issue batch operations asynchronously
        work = batch_op.issue(True)

        # Wait for completion before resetting
        work.wait()

        # Reset the work object
        work = None

        # Verify the results
        self._verify_results(recv_tensors, prev_rank)

    def _batch_sendrecv_input_deleted(self, count, dtype):
        """Test asynchronous batch SendRecv operations with input deleted after enqueue."""
        print(
            f"Testing async batch SendRecv with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to next rank and receives from previous rank
        next_rank = (self.rank + 1) % self.num_ranks
        prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create recv tensors that persist
        recv_tensors = []
        for _ in range(2):
            recv_tensor = self._create_recv_tensor(count, dtype)
            recv_tensors.append(recv_tensor)

        # Create batch operation object
        batch_op = self.torchcomm.batch_op_create()

        # Create send tensors and enqueue operations
        send_tensors = []
        for i in range(2):
            # Create send tensor with unique values
            send_tensor = self._create_send_tensor(count, dtype, i)
            send_tensors.append(send_tensor)
            batch_op.send(send_tensor, next_rank)

            # Add recv operations
            batch_op.recv(recv_tensors[i], prev_rank)

        # Issue batch operations synchronously
        batch_op.issue(False)

        # Delete the send tensors to simulate them going out of scope
        del send_tensors

        # Verify the results
        self._verify_results(recv_tensors, prev_rank)

    def _graph_batch_sendrecv(self, count, dtype):
        """Test CUDA Graph batch SendRecv operations."""
        print(
            f"Testing CUDA Graph batch SendRecv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Each rank sends to next rank and receives from previous rank
            next_rank = (self.rank + 1) % self.num_ranks
            prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

            # Create tensors for batch operations AFTER setting non-default stream but BEFORE graph capture
            send_tensors = []
            recv_tensors = []
            original_recv_tensors = []

            for i in range(2):
                # Create send tensor with unique values
                send_tensor = self._create_send_tensor(count, dtype, i)
                send_tensors.append(send_tensor)

                # Create recv tensor
                recv_tensor = self._create_recv_tensor(count, dtype)
                recv_tensors.append(recv_tensor)
                original_recv_tensors.append(recv_tensor.clone())

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the batch SendRecv operations in the graph
            with torch.cuda.graph(graph):
                # Create batch operation object
                batch_op = self.torchcomm.batch_op_create()

                for i in range(2):
                    batch_op.send(send_tensors[i], next_rank)
                    batch_op.recv(recv_tensors[i], prev_rank)

                # Issue batch operations synchronously
                batch_op.issue(False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output buffers before each replay
                for i, recv_tensor in enumerate(recv_tensors):
                    recv_tensor.copy_(original_recv_tensors[i])

                graph.replay()

                # Verify the results after each replay
                self._verify_results(recv_tensors, prev_rank)

    def _graph_batch_sendrecv_input_deleted(self, count, dtype):
        """Test CUDA Graph batch SendRecv operations with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph batch SendRecv with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Each rank sends to next rank and receives from previous rank
            next_rank = (self.rank + 1) % self.num_ranks
            prev_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

            # Create recv tensors that persist throughout the test
            recv_tensors = []
            original_recv_tensors = []
            for _ in range(2):
                recv_tensor = self._create_recv_tensor(count, dtype)
                recv_tensors.append(recv_tensor)
                original_recv_tensors.append(recv_tensor.clone())

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create send tensors for graph capture
            send_tensors = []
            for i in range(2):
                send_tensor = self._create_send_tensor(count, dtype, i)
                send_tensors.append(send_tensor)

            # Capture the batch SendRecv operations in the graph
            with torch.cuda.graph(graph):
                # Create batch operation object
                batch_op = self.torchcomm.batch_op_create()

                for i in range(2):
                    batch_op.send(send_tensors[i], next_rank)
                    batch_op.recv(recv_tensors[i], prev_rank)

                # Issue batch operations synchronously
                batch_op.issue(False)

            # Delete the send tensors to simulate them going out of scope
            del send_tensors

            # Replay the captured graph multiple times even though send tensors are deleted
            for _ in range(self.num_replays):
                # Reset output buffers before each replay
                for i, recv_tensor in enumerate(recv_tensors):
                    recv_tensor.copy_(original_recv_tensors[i])

                graph.replay()

                # Verify the results after each replay
                self._verify_results(recv_tensors, prev_rank)

    def _create_send_tensor(self, count, dtype, tensor_id):
        """Create send tensor with rank and tensor-specific values."""
        options = {"dtype": dtype, "device": self.device}
        value = (self.rank + 1) * 10 + tensor_id  # Make each tensor unique

        if dtype == torch.float or dtype == torch.bfloat16:
            return torch.ones(count, **options) * float(value)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(value)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(value % 128)
        return None

    def _create_recv_tensor(self, count, dtype):
        """Create receive tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count, **options)

    def _verify_results(self, recv_tensors, recv_rank):
        """Verify the results of the batch recv operations."""
        for i, recv_tensor in enumerate(recv_tensors):
            # Extract dtype and count from the tensor
            dtype = recv_tensor.dtype
            count = recv_tensor.numel()

            # Expected value for this tensor
            expected_value = (recv_rank + 1) * 10 + i

            if dtype == torch.float or dtype == torch.bfloat16:
                expected = torch.ones(count, dtype=dtype) * float(expected_value)
            elif dtype == torch.int:
                expected = torch.ones(count, dtype=dtype) * int(expected_value)
            elif dtype == torch.int8:
                expected_value = expected_value % 128
                expected = torch.ones(count, dtype=dtype) * int(expected_value)

            # Compare output with expected tensor
            description = f"recv rank {recv_rank} tensor {i}"
            if dtype == torch.float:
                self.assertTrue(
                    torch.allclose(recv_tensor.cpu(), expected),
                    f"Tensors not close enough for {description}",
                )
            else:
                self.assertTrue(
                    torch.equal(recv_tensor.cpu(), expected),
                    f"Tensors not equal for {description}",
                )

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Skipping gloo backend - batch operations not implemented",
    )
    def test_sync_batch_sendrecv(self):
        """Test synchronous batch SendRecv operations with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_batch_sendrecv(count, dtype)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Skipping gloo backend - batch operations not implemented",
    )
    def test_sync_batch_sendrecv_no_work(self):
        """Test synchronous batch SendRecv operations without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_batch_sendrecv_no_work(count, dtype)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Skipping gloo backend - batch operations not implemented",
    )
    def test_async_batch_sendrecv(self):
        """Test asynchronous batch SendRecv operations with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_batch_sendrecv(count, dtype)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Skipping gloo backend - batch operations not implemented",
    )
    def test_async_batch_sendrecv_early_reset(self):
        """Test asynchronous batch SendRecv operations with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_batch_sendrecv_early_reset(count, dtype)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Skipping gloo backend - batch operations not implemented",
    )
    def test_batch_sendrecv_input_deleted(self):
        """Test asynchronous batch SendRecv operations with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._batch_sendrecv_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
