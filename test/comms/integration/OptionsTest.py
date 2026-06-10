#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
from datetime import timedelta

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper

import torch
from torch.comms import ReduceOp


TENSOR_COUNT: int = 4
DEFAULT_HINTS_MAP = {"hint_key": "hint_value"}


class OptionsTest(unittest.TestCase):
    """
    Test class for options classes in all operations in TorchComm.

    This class verifies that the options classes are being accepted correctly as a Python argument.

    TODO: this test needs to be enhanced to verify that the option values were correctly plumbed,
    which will require a getter for the options via the work object or some similar method.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

        # Use device from torchcomm (supports both CUDA and CPU)
        self.device = self.torchcomm.get_device()

        options = {"dtype": torch.float, "device": self.device}

        # Prepare multiple tensors for all the operations to be tested.
        self.send_tensor = torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
        self.recv_tensor = torch.zeros(TENSOR_COUNT, **options)
        self.tensors_all_ranks = []
        for _ in range(self.num_ranks):
            self.tensors_all_ranks.append(torch.zeros(TENSOR_COUNT, **options))
        self.input_tensors = []
        for _ in range(self.num_ranks):
            self.input_tensors.append(
                torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
            )
        self.recv_tensor_single = torch.zeros(TENSOR_COUNT * self.num_ranks, **options)
        self.send_tensor_single = torch.zeros(TENSOR_COUNT * self.num_ranks, **options)

        # Create split sizes for all_to_all_v_single
        self.input_split_sizes = [TENSOR_COUNT] * self.num_ranks
        self.output_split_sizes = [TENSOR_COUNT] * self.num_ranks

        self.send_rank = (self.rank + 1) % self.num_ranks
        self.recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def test_send_recv(self):
        """Test Send and Recv operations with options."""
        if self.num_ranks < 2:
            self.skipTest("This test requires at least 2 ranks.")

        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            self.torchcomm.send(
                self.send_tensor,
                self.send_rank,
                False,
                hints=DEFAULT_HINTS_MAP,
                timeout=timedelta(seconds=600),
            )
            self.torchcomm.recv(
                self.recv_tensor,
                self.recv_rank,
                False,
                hints=DEFAULT_HINTS_MAP,
                timeout=timedelta(seconds=600),
            )
        else:
            # Odd ranks: receive first, then send
            self.torchcomm.recv(
                self.recv_tensor,
                self.recv_rank,
                False,
                hints=DEFAULT_HINTS_MAP,
                timeout=timedelta(seconds=600),
            )
            self.torchcomm.send(
                self.send_tensor,
                self.send_rank,
                False,
                hints=DEFAULT_HINTS_MAP,
                timeout=timedelta(seconds=600),
            )

    def test_all_reduce(self):
        """Test AllReduce operation with options."""
        self.torchcomm.all_reduce(
            self.send_tensor,
            ReduceOp.SUM,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_reduce(self):
        """Test Reduce operation with options."""
        self.torchcomm.reduce(
            self.send_tensor,
            0,
            ReduceOp.SUM,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_all_gather_single(self):
        """Test AllGatherSingle operation with options."""
        self.torchcomm.all_gather_single(
            self.recv_tensor_single,
            self.send_tensor,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_all_gather(self):
        """Test AllGather operation with options."""
        self.torchcomm.all_gather(
            self.tensors_all_ranks,
            self.send_tensor,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_gather(self):
        """Test Gather operation with options."""
        self.torchcomm.gather(
            self.tensors_all_ranks,
            self.send_tensor,
            0,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_reduce_scatter_single(self):
        """Test ReduceScatterSingle operation with options."""
        self.torchcomm.reduce_scatter_single(
            self.recv_tensor,
            self.send_tensor_single,
            ReduceOp.SUM,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_reduce_scatter(self):
        """Test ReduceScatter operation with options."""
        self.torchcomm.reduce_scatter(
            self.recv_tensor,
            self.tensors_all_ranks,
            ReduceOp.SUM,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_scatter(self):
        """Test Scatter operation with options."""
        self.torchcomm.scatter(
            self.recv_tensor,
            self.tensors_all_ranks,
            0,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_all_to_all(self):
        """Test AllToAll operation with options."""
        self.torchcomm.all_to_all(
            self.tensors_all_ranks,
            self.input_tensors,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_all_to_all_single(self):
        """Test AllToAllSingle operation with options."""
        self.torchcomm.all_to_all_single(
            self.recv_tensor_single,
            self.send_tensor_single,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_all_to_all_v_single(self):
        """Test AllToAllVSingle operation with options."""
        self.torchcomm.all_to_all_v_single(
            self.recv_tensor_single,
            self.send_tensor_single,
            self.output_split_sizes,
            self.input_split_sizes,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_broadcast(self):
        """Test Broadcast operation with options."""
        self.torchcomm.broadcast(
            self.send_tensor,
            0,
            False,
            hints=DEFAULT_HINTS_MAP,
            timeout=timedelta(seconds=600),
        )

    def test_barrier(self):
        """Test Barrier operation with options."""
        work = self.torchcomm.barrier(
            False, hints=DEFAULT_HINTS_MAP, timeout=timedelta(seconds=600)
        )
        work.wait()


if __name__ == "__main__":
    unittest.main()
