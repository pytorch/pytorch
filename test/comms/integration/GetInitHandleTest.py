#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for the get_init_handle() Fault Tolerance API in TorchComms.

This test verifies:
1. Backends that don't implement get_init_handle() raise RuntimeError
2. Backends that implement get_init_handle() return valid init handles
"""

import os
import unittest

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper


class GetInitHandleTest(unittest.TestCase):
    """Test class for get_init_handle() fault tolerance API."""

    # Backends that implement get_init_handle()
    SUPPORTED_BACKENDS = {"mccl", "gloo", "nccl"}

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
        """Check if current backend supports get_init_handle()."""
        return self.backend in self.SUPPORTED_BACKENDS

    def test_get_init_handle(self):
        """Test get_init_handle() behavior based on backend support."""
        if self._is_supported_backend():
            # Backend supports get_init_handle() - should return valid handle
            init_handle = self.torchcomm.get_init_handle()

            self.assertIsInstance(init_handle, str)
            self.assertNotEqual(init_handle, "")
            self.assertGreater(len(init_handle), 0)

            print(f"[Rank {self.rank}] Init handle: {init_handle}")
        else:
            # Backend doesn't support get_init_handle() - should raise RuntimeError
            with self.assertRaises(RuntimeError) as context:
                self.torchcomm.get_init_handle()

            self.assertIn("getInitHandle not implemented", str(context.exception))
            print(
                f"[Rank {self.rank}] Expected RuntimeError raised: {context.exception}"
            )

    def test_get_init_handle_consistent(self):
        """Test that get_init_handle() returns consistent values on multiple calls."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support get_init_handle()")

        handle1 = self.torchcomm.get_init_handle()
        handle2 = self.torchcomm.get_init_handle()

        self.assertEqual(handle1, handle2)
        print(f"[Rank {self.rank}] Init handle is consistent: {handle1}")


if __name__ == "__main__":
    unittest.main()
