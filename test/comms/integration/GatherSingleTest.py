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


class GatherSingleTest(unittest.TestCase):
    """Test class for gather_single operations in TorchComm."""

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
        self.torchcomm = None
        self.wrapper = None

    def _sync_gather_single(self, count, dtype):
        """Test synchronous gather_single with work object."""
        print(
            f"Testing sync gather_single with count={count} and dtype={get_dtype_name(dtype)}"
        )

        input_tensor = self._create_input_tensor(count, dtype)
        root_rank = 0
        output_tensor = self._create_output_tensor(root_rank, count, dtype)

        work = self.torchcomm.gather_single(
            output_tensor, input_tensor, root_rank, False
        )
        work.wait()

        self._verify_results(output_tensor, root_rank)

    def _sync_gather_single_no_work(self, count, dtype):
        """Test synchronous gather_single without work object."""
        print(
            f"Testing sync gather_single without work object with count={count} and dtype={get_dtype_name(dtype)}"
        )

        input_tensor = self._create_input_tensor(count, dtype)
        root_rank = 0
        output_tensor = self._create_output_tensor(root_rank, count, dtype)

        self.torchcomm.gather_single(output_tensor, input_tensor, root_rank, False)

        self._verify_results(output_tensor, root_rank)

    def _async_gather_single(self, count, dtype):
        """Test asynchronous gather_single with wait."""
        print(
            f"Testing async gather_single with count={count} and dtype={get_dtype_name(dtype)}"
        )

        input_tensor = self._create_input_tensor(count, dtype)
        root_rank = 0
        output_tensor = self._create_output_tensor(root_rank, count, dtype)

        work = self.torchcomm.gather_single(
            output_tensor, input_tensor, root_rank, True
        )
        work.wait()

        self._verify_results(output_tensor, root_rank)

    def _async_gather_single_early_reset(self, count, dtype):
        """Test asynchronous gather_single with early reset."""
        print(
            f"Testing async gather_single with early reset with count={count} and dtype={get_dtype_name(dtype)}"
        )

        input_tensor = self._create_input_tensor(count, dtype)
        root_rank = 0
        output_tensor = self._create_output_tensor(root_rank, count, dtype)

        work = self.torchcomm.gather_single(
            output_tensor, input_tensor, root_rank, True
        )
        work.wait()
        work = None

        self._verify_results(output_tensor, root_rank)

    def _gather_single_input_deleted(self, count, dtype):
        """Test gather_single with input deleted after enqueue."""
        print(
            f"Testing gather_single with input deleted after enqueue with count={count} and dtype={get_dtype_name(dtype)}"
        )

        root_rank = 0
        output_tensor = self._create_output_tensor(root_rank, count, dtype)
        input_tensor = self._create_input_tensor(count, dtype)

        self.torchcomm.gather_single(output_tensor, input_tensor, root_rank, False)
        del input_tensor

        self._verify_results(output_tensor, root_rank)

    def _graph_gather_single(self, count, dtype):
        """Test CUDA Graph gather_single."""
        print(
            f"Testing CUDA Graph gather_single with count={count} and dtype={get_dtype_name(dtype)}"
        )

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            input_tensor = self._create_input_tensor(count, dtype)
            root_rank = 0
            output_tensor = self._create_output_tensor(root_rank, count, dtype)
            original_output_tensor = output_tensor.clone()

            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                self.torchcomm.gather_single(
                    output_tensor, input_tensor, root_rank, False
                )

            for _ in range(self.num_replays):
                output_tensor.copy_(original_output_tensor)
                graph.replay()
                self._verify_results(output_tensor, root_rank)

    def _graph_gather_single_input_deleted(self, count, dtype):
        """Test CUDA Graph gather_single with input deleted after graph creation."""
        print(
            f"Testing CUDA Graph gather_single with input deleted after graph creation with count={count} and dtype={get_dtype_name(dtype)}"
        )

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            root_rank = 0
            output_tensor = self._create_output_tensor(root_rank, count, dtype)
            original_output_tensor = output_tensor.clone()

            graph = torch.cuda.CUDAGraph()
            input_tensor = self._create_input_tensor(count, dtype)

            with torch.cuda.graph(graph):
                self.torchcomm.gather_single(
                    output_tensor, input_tensor, root_rank, False
                )

            del input_tensor

            for _ in range(self.num_replays):
                if self.rank == root_rank:
                    output_tensor.copy_(original_output_tensor)
                graph.replay()
                self._verify_results(output_tensor, root_rank)

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

    def _create_output_tensor(self, root_rank, count, dtype):
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count * self.num_ranks, **options)

    def synchronize_stream(self):
        """Synchronize the current stream."""
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    def _verify_results(self, output_tensor, root_rank):
        """Verify the results of the gather_single operation."""
        if self.rank != root_rank:
            self.synchronize_stream()
            return

        count = output_tensor.numel() // self.num_ranks

        for i in range(self.num_ranks):
            section = output_tensor[i * count : (i + 1) * count]
            dtype = section.dtype

            if dtype == torch.float:
                expected = torch.ones(count, dtype=dtype) * float(i + 1)
                self.assertTrue(
                    torch.allclose(section.cpu(), expected),
                    f"Tensors not close enough for gather_single rank {i} section",
                )
            else:
                expected = torch.ones(count, dtype=dtype) * int(i + 1)
                self.assertTrue(
                    torch.equal(section.cpu(), expected),
                    f"Tensors not equal for gather_single rank {i} section",
                )

    def test_sync_gather_single(self):
        """Test synchronous gather_single with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_gather_single(count, dtype)

    def test_sync_gather_single_no_work(self):
        """Test synchronous gather_single without work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_gather_single_no_work(count, dtype)

    def test_async_gather_single(self):
        """Test asynchronous gather_single with wait."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_gather_single(count, dtype)

    def test_async_gather_single_early_reset(self):
        """Test asynchronous gather_single with early reset."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._async_gather_single_early_reset(count, dtype)

    def test_gather_single_input_deleted(self):
        """Test gather_single with input deleted after enqueue."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._gather_single_input_deleted(count, dtype)


if __name__ == "__main__":
    unittest.main()
