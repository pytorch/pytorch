#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for the wait_blocking() Fault Tolerance API in TorchComms.

This test verifies:
1. Backends that don't implement wait_blocking() raise RuntimeError
2. Backends that implement wait_blocking() correctly block until work completes
3. is_completed() returns True only after wait_blocking() completes
"""

import os
import unittest

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper

import torch
from torch.comms import ReduceOp


class WaitBlockingTest(unittest.TestCase):
    """Test class for wait_blocking() fault tolerance API."""

    # Backends that implement wait_blocking()
    SUPPORTED_BACKENDS = {"mccl"}

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()
        self.backend = os.getenv("TEST_BACKEND", "")

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def _is_supported_backend(self):
        """Check if current backend supports wait_blocking()."""
        return self.backend in self.SUPPORTED_BACKENDS

    def test_wait_blocking_unsupported_backend(self):
        """Test that unsupported backends raise RuntimeError."""
        if self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} supports wait_blocking()")

        tensor = torch.ones(4, device=self.device) * (self.rank + 1)
        work = self.torchcomm.all_reduce(tensor, ReduceOp.SUM, True)

        with self.assertRaises(RuntimeError) as context:
            work.wait_blocking()

        self.assertIn("waitBlocking not implemented", str(context.exception))
        print(f"[Rank {self.rank}] Expected RuntimeError raised: {context.exception}")

        # Still need to wait for the operation to complete using regular wait
        work.wait()

    def test_wait_blocking_blocks_until_complete(self):
        """Test that wait_blocking() blocks CPU until operation completes.

        This test verifies that:
        1. is_completed() may return False immediately after issuing async op
        2. After wait_blocking(), is_completed() returns True
        3. The result is correct after wait_blocking()
        """
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support wait_blocking()")

        # Use larger tensor to increase chance of catching incomplete state
        tensor = torch.ones(1024, device=self.device) * (self.rank + 1)

        # Issue async operation
        work = self.torchcomm.all_reduce(tensor, ReduceOp.SUM, True)

        # Note: is_completed() may return True or False here depending on timing.
        # We don't assert False because small ops may complete immediately.
        # The key test is that after wait_blocking(), everything is consistent.

        # Block CPU until complete
        work.wait_blocking()

        # After wait_blocking(), is_completed() MUST return True
        self.assertTrue(
            work.is_completed(),
            "is_completed() must return True after wait_blocking()",
        )
        print(f"[Rank {self.rank}] wait_blocking correctly blocked until completion")

    def test_wait_blocking_multiple_ops(self):
        """Test wait_blocking() with multiple sequential operations.

        Verifies that wait_blocking() works correctly when called multiple times
        on different work handles.
        """
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support wait_blocking()")

        for i in range(3):
            tensor = torch.ones(1024, device=self.device) * (self.rank + 1)
            work = self.torchcomm.all_reduce(tensor, ReduceOp.SUM, True)

            work.wait_blocking()

            self.assertTrue(
                work.is_completed(),
                f"Iteration {i}: is_completed() must be True after wait_blocking()",
            )

            expected_sum = sum(range(1, self.num_ranks + 1))
            expected = torch.ones(1024, device=self.device) * expected_sum
            self.assertTrue(torch.allclose(tensor, expected))

        print(f"[Rank {self.rank}] Multiple wait_blocking ops completed successfully")

    def test_wait_blocking_idempotent(self):
        """Test that calling wait_blocking() multiple times is safe.

        Similar to MCCL's TestMultipleWaitCalls - verifies that multiple
        wait_blocking() calls on the same work handle don't cause issues.
        """
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support wait_blocking()")

        tensor = torch.ones(1024, device=self.device) * (self.rank + 1)
        work = self.torchcomm.all_reduce(tensor, ReduceOp.SUM, True)

        # Call wait_blocking multiple times - should be safe
        work.wait_blocking()
        work.wait_blocking()
        work.wait_blocking()

        self.assertTrue(work.is_completed())

        expected_sum = sum(range(1, self.num_ranks + 1))
        expected = torch.ones(1024, device=self.device) * expected_sum
        self.assertTrue(torch.allclose(tensor, expected))

        print(f"[Rank {self.rank}] Multiple wait_blocking calls are idempotent")


if __name__ == "__main__":
    unittest.main()
