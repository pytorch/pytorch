#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)

import torch


class SendRecvTest(unittest.TestCase):
    """Test class for send/recv operations in TorchComm."""

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

    def _sync_send_recv(self, count, dtype):
        """Test synchronous send/recv with work object."""
        print(
            f"Testing sync send/recv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to the next rank and receives from the previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create input tensor with rank-specific values
        send_tensor = self._create_send_tensor(count, dtype)

        # Create output tensor to receive data
        recv_tensor = self._create_recv_tensor(count, dtype)

        # Alternate send/recv order based on rank to avoid deadlock
        # Even ranks send first, then receive
        # Odd ranks receive first, then send
        send_work = None
        recv_work = None

        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            send_work = self.torchcomm.send(send_tensor, send_rank, False)
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, False)
        else:
            # Odd ranks: receive first, then send
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, False)
            send_work = self.torchcomm.send(send_tensor, send_rank, False)
        send_work.wait()
        recv_work.wait()

        # Verify the results
        self._verify_results(recv_tensor, recv_rank)

    def _sync_send_recv_no_work(self, count, dtype):
        """Test synchronous send/recv without work object."""
        print(
            f"Testing sync send/recv without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to the next rank and receives from the previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create input tensor with rank-specific values
        send_tensor = self._create_send_tensor(count, dtype)

        # Create output tensor to receive data
        recv_tensor = self._create_recv_tensor(count, dtype)

        # Alternate send/recv order based on rank to avoid deadlock
        # Even ranks send first, then receive
        # Odd ranks receive first, then send
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            self.torchcomm.send(send_tensor, send_rank, False)
            self.torchcomm.recv(recv_tensor, recv_rank, False)
        else:
            # Odd ranks: receive first, then send
            self.torchcomm.recv(recv_tensor, recv_rank, False)
            self.torchcomm.send(send_tensor, send_rank, False)

        # Verify the results
        self._verify_results(recv_tensor, recv_rank)

    def _async_send_recv(self, count, dtype):
        """Test asynchronous send/recv with wait."""
        print(
            f"Testing async send/recv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to the next rank and receives from the previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create input tensor with rank-specific values
        send_tensor = self._create_send_tensor(count, dtype)

        # Create output tensor to receive data
        recv_tensor = self._create_recv_tensor(count, dtype)

        # Alternate send/recv order based on rank to avoid deadlock
        # Even ranks send first, then receive
        # Odd ranks receive first, then send
        send_work = None
        recv_work = None

        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            send_work = self.torchcomm.send(send_tensor, send_rank, True)
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, True)
        else:
            # Odd ranks: receive first, then send
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, True)
            send_work = self.torchcomm.send(send_tensor, send_rank, True)

        # Wait for the operations to complete
        # For async operations, we can wait in any order
        send_work.wait()
        recv_work.wait()

        # Verify the results
        self._verify_results(recv_tensor, recv_rank)

    def _async_send_recv_early_reset(self, count, dtype):
        """Test asynchronous send/recv with early reset."""
        print(
            f"Testing async send/recv with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to the next rank and receives from the previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create input tensor with rank-specific values
        send_tensor = self._create_send_tensor(count, dtype)

        # Create output tensor to receive data
        recv_tensor = self._create_recv_tensor(count, dtype)

        # Alternate send/recv order based on rank to avoid deadlock
        # Even ranks send first, then receive
        # Odd ranks receive first, then send
        send_work = None
        recv_work = None

        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            send_work = self.torchcomm.send(send_tensor, send_rank, True)
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, True)
        else:
            # Odd ranks: receive first, then send
            recv_work = self.torchcomm.recv(recv_tensor, recv_rank, True)
            send_work = self.torchcomm.send(send_tensor, send_rank, True)

        # Wait for the operations to complete before resetting
        send_work.wait()
        recv_work.wait()

        # Reset the work objects
        send_work = None
        recv_work = None

        # Verify the results
        self._verify_results(recv_tensor, recv_rank)

    def _send_recv_input_deleted(self, count, dtype):
        """Test asynchronous send/recv with input deleted after enqueue."""
        print(
            f"Testing async send/recv with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends to the next rank and receives from the previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Create recv tensor that persists
        recv_tensor = self._create_recv_tensor(count, dtype)

        # Create send tensor and enqueue operations
        send_tensor = self._create_send_tensor(count, dtype)

        # Alternate send/recv order based on rank to avoid deadlock
        # Even ranks send first, then receive
        # Odd ranks receive first, then send
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            self.torchcomm.send(send_tensor, send_rank, False)
            self.torchcomm.recv(recv_tensor, recv_rank, False)
        else:
            # Odd ranks: receive first, then send
            self.torchcomm.recv(recv_tensor, recv_rank, False)
            self.torchcomm.send(send_tensor, send_rank, False)

        # Delete the send tensor to simulate it going out of scope
        del send_tensor

        # Verify the results
        self._verify_results(recv_tensor, recv_rank)

    def _graph_send_recv(self, count, dtype):
        """Test CUDA Graph send/recv."""
        print(
            f"Testing CUDA Graph send/recv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create send/recv parameters AFTER setting non-default stream but BEFORE graph capture
            send_rank = (self.rank + 1) % self.num_ranks
            recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks
            send_tensor = self._create_send_tensor(count, dtype)
            recv_tensor = self._create_recv_tensor(count, dtype)
            original_recv_tensor = recv_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the send/recv operations in the graph
            with torch.cuda.graph(graph):
                # Call send/recv without keeping the work objects
                # Alternate send/recv order based on rank to avoid deadlock
                if self.rank % 2 == 0:
                    # Even ranks: send first, then receive
                    self.torchcomm.send(send_tensor, send_rank, False)
                    self.torchcomm.recv(recv_tensor, recv_rank, False)
                else:
                    # Odd ranks: receive first, then send
                    self.torchcomm.recv(recv_tensor, recv_rank, False)
                    self.torchcomm.send(send_tensor, send_rank, False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                # Reset output buffer before each replay
                recv_tensor.copy_(original_recv_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(recv_tensor, recv_rank)

    def _graph_send_recv_input_deleted(self, count, dtype):
        """Test CUDA Graph send/recv with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph send/recv with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Each rank sends to the next rank and receives from the previous rank
            send_rank = (self.rank + 1) % self.num_ranks
            recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

            # Create recv tensor that persists throughout the test
            recv_tensor = self._create_recv_tensor(count, dtype)
            original_recv_tensor = recv_tensor.clone()

            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Create send tensor for graph capture
            send_tensor = self._create_send_tensor(count, dtype)

            # Capture the send/recv operations in the graph
            with torch.cuda.graph(graph):
                # Call send/recv without keeping the work objects
                # Alternate send/recv order based on rank to avoid deadlock
                if self.rank % 2 == 0:
                    # Even ranks: send first, then receive
                    self.torchcomm.send(send_tensor, send_rank, False)
                    self.torchcomm.recv(recv_tensor, recv_rank, False)
                else:
                    # Odd ranks: receive first, then send
                    self.torchcomm.recv(recv_tensor, recv_rank, False)
                    self.torchcomm.send(send_tensor, send_rank, False)

            # Delete the send tensor to simulate it going out of scope
            del send_tensor

            # Replay the captured graph multiple times even though send tensor is deleted
            for _ in range(self.num_replays):
                # Reset output buffer before each replay
                recv_tensor.copy_(original_recv_tensor)

                graph.replay()

                # Verify the results after each replay
                self._verify_results(recv_tensor, recv_rank)

    def _create_send_tensor(self, count, dtype):
        """Create send tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if dtype == torch.float:
            return torch.ones(count, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(self.rank + 1)
        return None

    def _create_recv_tensor(self, count, dtype):
        """Create receive tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count, **options)

    def _verify_results(self, recv_tensor, recv_rank):
        """Verify the results of the recv operation."""
        # Extract dtype and count from the tensor
        dtype = recv_tensor.dtype
        count = recv_tensor.numel()

        # Expected value for this tensor
        if dtype == torch.float:
            expected = torch.ones(count, dtype=dtype) * float(recv_rank + 1)
        elif dtype == torch.int:
            expected = torch.ones(count, dtype=dtype) * int(recv_rank + 1)
        elif dtype == torch.int8:
            expected = torch.ones(count, dtype=dtype) * int(recv_rank + 1)

        # Compare output with expected tensor
        description = f"recv rank {recv_rank} tensor"
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

    def test_sync_send_recv(self):
        """Test synchronous send/recv with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_send_recv(count, dtype)

    def test_sync_send_recv_no_work(self):
        """Test synchronous send/recv without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_send_recv_no_work(count, dtype)

    def test_async_send_recv(self):
        """Test asynchronous send/recv with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_send_recv(count, dtype)

    def test_async_send_recv_early_reset(self):
        """Test asynchronous send/recv with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_send_recv_early_reset(count, dtype)

    def test_send_recv_input_deleted(self):
        """Test asynchronous send/recv with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._send_recv_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
