#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torch.comms


class MemPoolTorchCommTest(unittest.TestCase):
    def setUp(self) -> None:
        if "TEST_BACKEND" not in os.environ:
            raise AssertionError("TEST_BACKEND not set")
        self.device_ = torch.device("cuda")
        self.backend_ = os.environ["TEST_BACKEND"]

        self.num_comms_ = 16
        self.tensor_size_ = 1024 * 1024
        self.comms_ = []
        for x in range(self.num_comms_):
            self.comms_.append(
                torch.comms.new_comm(
                    self.backend_, self.device_, name=f"test_mem_pool_{x}"
                )
            )
        # Get allocator using global function - can be obtained once and reused
        # This should be called after torch.comms.new_comm(), since the mempool
        # related symbols is lazy loaded at runtime during torch.comms.new_comm()
        self.allocator_ = torch.comms.get_mem_allocator(self.backend_)

    def tearDown(self) -> None:
        for comm in self.comms_:
            comm.finalize()

    def _create_input_tensor(self, comm):
        """Create input tensor with rank-specific values."""
        rank = comm.get_rank()
        return torch.ones(self.tensor_size_, device=self.device_) * float(rank + 1)

    def _verify_results(self, tensor, comm):
        """Verify the results of the all_reduce operation."""
        num_ranks = comm.get_size()
        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(tensor.cpu(), float(expected))
        torch.testing.assert_close(tensor.cpu(), expected_tensor)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") not in ["nccl"]
        or torch.cuda.get_device_capability() < (9, 0),
        "Skipping NCCL-only mem pool tests",
    )
    def test_mem_pool(self) -> None:
        # Use the global allocator obtained in setUp - shared across all comms
        tensors = []
        for comm in self.comms_:
            pool = torch.cuda.MemPool(self.allocator_)
            with torch.cuda.use_mem_pool(pool):
                tensor = self._create_input_tensor(comm)
            tensors.append(tensor)
            comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, False).wait()

        torch.cuda.current_stream().synchronize()

        for i, comm in enumerate(self.comms_):
            self._verify_results(tensors[i], comm)


if __name__ == "__main__":
    unittest.main()
