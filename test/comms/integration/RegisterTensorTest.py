#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

from integration.helpers.TorchCommTestHelpers import skip_backend, TorchCommTestWrapper

import torch


class RegisterTensorTest(unittest.TestCase):
    """Test tensor_register/tensor_deregister via CPU broadcast.

    Verifies that pre-registering a CPU tensor with the communicator
    enables zero-copy RDMA for CPU collectives. Backends that override
    tensor_register should inherit this test.
    """

    counts = [4, 1024, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8]

    def setUp(self):
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def _verify_broadcast_results(self, tensor, expected_value, count, dtype):
        expected = torch.ones(count, dtype=dtype, device="cpu") * expected_value
        if dtype == torch.float:
            self.assertTrue(
                torch.allclose(tensor, expected),
                f"CPU broadcast tensors not close enough for count={count}",
            )
        else:
            self.assertTrue(
                torch.equal(tensor, expected),
                f"CPU broadcast tensors not equal for count={count}",
            )

    def _cpu_broadcast_with_tensor_register(self, count, dtype):
        """Test CPU broadcast using TorchComm.tensor_register() public API."""
        root_rank = 0
        root_value = 99

        if self.rank == root_rank:
            tensor = torch.ones(count, dtype=dtype, device="cpu") * root_value
        else:
            tensor = torch.zeros(count, dtype=dtype, device="cpu")

        self.torchcomm.tensor_register(tensor)

        try:
            work = self.torchcomm.broadcast(tensor, root_rank, False)
            self.assertTrue(work.is_completed())
            self._verify_broadcast_results(tensor, root_value, count, dtype)
        finally:
            self.torchcomm.tensor_deregister(tensor)

    @skip_backend("nccl", msg="tensor_register not implemented for backend: ")
    @skip_backend("gloo", msg="tensor_register not implemented for backend: ")
    def test_cpu_broadcast_with_tensor_register(self):
        """Test CPU broadcast with TorchComm.tensor_register() public API."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._cpu_broadcast_with_tensor_register(count, dtype)


if __name__ == "__main__":
    unittest.main()
