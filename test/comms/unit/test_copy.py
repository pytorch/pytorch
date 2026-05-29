#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os
import sys


# Make test/comms importable so `helpers` / `integration` resolve when this
# file is run directly (run_test.py runs `python comms/unit/<file>.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper

import torch
from torch._C._comms import ReduceOp
from torch.comms import new_comm
from torch.testing._internal.common_utils import run_tests, TestCase


class TestReduceOpCopy(TestCase):
    """Tests for ReduceOp copy/deepcopy."""

    def test_reduceop_copy(self):
        """Test that ReduceOp copy creates a new object with same value."""
        op = ReduceOp.SUM
        op_copy = copy.copy(op)

        self.assertEqual(op.type, op_copy.type)

    def test_reduceop_deepcopy(self):
        """Test that ReduceOp deepcopy creates a new object and updates memo."""
        op = ReduceOp.SUM
        memo = {}
        op_copy = copy.deepcopy(op, memo)

        self.assertEqual(op.type, op_copy.type)
        self.assertIn(id(op), memo)

    def test_reduceop_deepcopy_memo_first(self):
        """Test that ReduceOp deepcopy returns memoized object if present."""
        op = ReduceOp.SUM

        sentinel = object()
        memo = {id(op): sentinel}

        result = copy.deepcopy(op, memo)
        self.assertIs(result, sentinel)


class TestCommCopy(TestCase):
    """Tests for TorchComm copy/deepcopy using the dummy backend."""

    def setUp(self):
        """Set up test environment before each test."""
        self.torchcomm = new_comm("fake", torch.device("cpu"), name="test_copy_comm")

    def tearDown(self):
        """Clean up after each test."""
        if self.torchcomm:
            self.torchcomm.finalize()
        self.torchcomm = None

    def test_comm_copy(self):
        """Test that TorchComm copy returns the same object."""
        comm_copy = copy.copy(self.torchcomm)
        self.assertIs(self.torchcomm, comm_copy)

    def test_comm_deepcopy(self):
        """Test that TorchComm deepcopy returns the same object and updates memo."""
        memo = {}
        comm_copy = copy.deepcopy(self.torchcomm, memo)

        self.assertIs(self.torchcomm, comm_copy)
        self.assertIn(id(self.torchcomm), memo)

    def test_comm_deepcopy_memo_first(self):
        """Test that TorchComm deepcopy returns memoized object if present."""
        sentinel = object()
        memo = {id(self.torchcomm): sentinel}

        result = copy.deepcopy(self.torchcomm, memo)
        self.assertIs(result, sentinel)


class TestWindowCopy(TestCase):
    """Tests for TorchCommWindow copy/deepcopy.

    These tests require a backend that supports windows (e.g., nccl).
    """

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        try:
            self.wrapper = self.get_wrapper()
            self.torchcomm = self.wrapper.get_torchcomm()
            self.rank = self.torchcomm.get_rank()
            self.num_ranks = self.torchcomm.get_size()
            self.device = self.torchcomm.get_device()
            self.window = self.torchcomm.new_window()
        except RuntimeError:
            self.skipTest(
                f"Unable to create window using backend {os.getenv('TEST_BACKEND')}."
            )

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def test_window_copy(self):
        """Test that TorchCommWindow copy returns the same object."""
        window_copy = copy.copy(self.window)

        self.assertIs(self.window, window_copy)

    def test_window_deepcopy(self):
        """Test that TorchCommWindow deepcopy creates a new window."""
        memo = {}
        window_copy = copy.deepcopy(self.window, memo)

        self.assertIsNot(self.window, window_copy)
        self.assertIn(id(self.window), memo)

    def test_window_deepcopy_with_tensor(self):
        """Test that TorchCommWindow deepcopy clones the registered tensor."""
        device = self.torchcomm.get_device()
        tensor = torch.zeros(10, dtype=torch.float32, device=device)
        self.window.tensor_register(tensor)

        memo = {}
        window_copy = copy.deepcopy(self.window, memo)

        self.assertIsNot(self.window, window_copy)
        self.assertIn(id(self.window), memo)

        cloned_tensor = window_copy.get_tensor()
        self.assertIsNotNone(cloned_tensor)

        self.assertIsNot(tensor, cloned_tensor)
        self.assertTrue(torch.equal(tensor, cloned_tensor))

        tensor[0] = 42.0
        self.assertNotEqual(tensor[0].item(), cloned_tensor[0].item())

        self.window.tensor_deregister()
        window_copy.tensor_deregister()

    def test_window_deepcopy_memo_first(self):
        """Test that TorchCommWindow deepcopy returns memoized object if present."""
        sentinel = object()
        memo = {id(self.window): sentinel}

        result = copy.deepcopy(self.window, memo)
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    run_tests()
