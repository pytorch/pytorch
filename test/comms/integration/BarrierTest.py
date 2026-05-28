#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper

import torch


class BarrierTest(unittest.TestCase):
    """Test class for barrier operations in TorchComm."""

    # Class variables for test parameters
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

    def _sync_barrier(self):
        """Test synchronous barrier with work object."""
        print("Testing sync barrier")

        # Call barrier
        work = self.torchcomm.barrier(False)
        work.wait()

        # No explicit verification needed for barrier, just ensure it completes

    def _sync_barrier_no_work(self):
        """Test synchronous barrier without work object."""
        print("Testing sync barrier without work object")

        # Call barrier without keeping the work object
        self.torchcomm.barrier(False)

    def _async_barrier(self):
        """Test asynchronous barrier with wait."""
        print("Testing async barrier")

        # Call barrier
        work = self.torchcomm.barrier(True)

        # Wait for the barrier to complete
        work.wait()

    def _async_barrier_early_reset(self):
        """Test asynchronous barrier with early reset."""
        print("Testing async barrier with early reset")

        # Call barrier
        work = self.torchcomm.barrier(True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

    def _graph_barrier(self):
        """Test CUDA Graph barrier."""
        print("Testing CUDA Graph barrier")

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the barrier operation in the graph
            with torch.cuda.graph(graph):
                # Call barrier without keeping the work object
                self.torchcomm.barrier(False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                graph.replay()

                # No explicit verification needed for barrier, just ensure it completes

    def test_sync_barrier(self):
        """Test synchronous barrier with work object."""
        self._sync_barrier()

    def test_sync_barrier_no_work(self):
        """Test synchronous barrier without work object."""
        self._sync_barrier_no_work()

    def test_async_barrier(self):
        """Test asynchronous barrier with wait."""
        self._async_barrier()

    def test_async_barrier_early_reset(self):
        """Test asynchronous barrier with early reset."""
        self._async_barrier_early_reset()


if __name__ == "__main__":
    unittest.main()
