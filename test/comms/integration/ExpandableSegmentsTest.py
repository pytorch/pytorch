#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Check that large allocations that span multiple physical caching allocator segment
chunks are correctly registered. This is the behavior when expandable segments is set.
"""

import os
import unittest


os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALLOC")
os.environ.setdefault("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal")
os.environ.setdefault("NCCL_ALLREDUCE_ALGO", "ctran")

from integration.helpers.TorchCommTestHelpers import (
    get_rank_and_size,
    TorchCommTestWrapper,
)

import torch
from torch.comms import ReduceOp


class ExpandableSegmentsTest(unittest.TestCase):
    """Test class for expandable segments multi-segment registration.

    NCCL_CTRAN_REGISTER (async/eager) is configured via Buck target configurations.
    """

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def get_device(self) -> torch.device:
        """Get device for this rank without creating a comm."""
        rank, _ = get_rank_and_size()
        if device_str := os.environ.get("TEST_DEVICE"):
            return torch.device(device_str)

        if torch.accelerator.is_available():
            device_count = torch.accelerator.device_count()
            if device_count > 0:
                device_id = rank % device_count
                accelerator = torch.accelerator.current_accelerator()
                assert accelerator is not None
                device_type = accelerator.type
                return torch.device(f"{device_type}:{device_id}")
        return torch.device("cpu")

    def setUp(self):
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def test_large_allocation(self):
        """
        Test that large allocations spanning multiple physical memory allocations work correctly.

        With expandable segments enabled, PyTorch allocates memory in 20MB segment chunks.
        A 110MB allocation should span 6 physical 20 MB segment chunks. This test verifies that:
        1. The CCA hook receives the SEGMENT_MAP event for the full range
        2. ctran's pinRange() discovers all underlying physical segments and caches them
        3. At collective time, the elements of the segment cache backing the input tensor are
           correctly registered with NCCL
        """
        count = 110 * 1024 * 1024
        dtype = torch.uint8

        # Create input tensor - this triggers SEGMENT_MAP for a ~110MB allocation
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * float(
            self.rank + 1
        )

        # Perform all_reduce - this exercises the full ctran registration path
        work = self.torchcomm.all_reduce(input_tensor, ReduceOp.SUM, False)
        work.wait()

        # Check collective result
        expected = self.num_ranks * (self.num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), float(expected))
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    def test_collective_after_empty_cache(self):
        """Test that allocations and collectives work correctly after empty_cache.

        This test verifies that the CCA hook correctly handles the case where
        PyTorch fires SEGMENT_MAP for a new allocation before firing SEGMENT_UNMAP
        for old memory at the same address (which can happen when addresses are
        reused after empty_cache).
        """
        count = 110 * 1024 * 1024
        dtype = torch.uint8

        # First allocation and collective
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * float(
            self.rank + 1
        )
        work = self.torchcomm.all_reduce(input_tensor, ReduceOp.SUM, False)
        work.wait()

        expected = self.num_ranks * (self.num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), float(expected))
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

        # Delete tensor and call empty_cache to trigger SEGMENT_UNMAP
        del input_tensor
        torch.cuda.empty_cache()

        # Second allocation - may reuse same address, triggering SEGMENT_MAP
        # before the SEGMENT_UNMAP from above is processed
        input_tensor2 = torch.ones(count, dtype=dtype, device=self.device) * float(
            self.rank + 1
        )
        work2 = self.torchcomm.all_reduce(input_tensor2, ReduceOp.SUM, False)
        work2.wait()

        print(
            f"rank {self.rank} input_tensor2: {input_tensor2}, expected: {expected_tensor}"
        )
        torch.testing.assert_close(input_tensor2.cpu(), expected_tensor)

    def test_memory_allocated_before_comm_creation(self):
        """Test that memory allocated BEFORE the CCA hook is attached gets registered.

        This test verifies the registerMemPreHook path in TorchCommNCCLCCA.cpp:
        1. When the CCA trace hook is first attached, registerMemPreHook() is called
        2. It queries CUDACachingAllocator::snapshot() to find all existing allocations
        3. Each segment is registered globally via global_register_address()
        4. The device is auto-detected from the buffer pointer by ctran

        The test flow is:
        1. Clean up the existing comm from setUp()
        2. Allocate memory BEFORE creating a new comm
        3. Create a new comm (the CCA hook attaches and registers pre-existing memory)
        4. Use the pre-allocated memory in a collective to verify registration worked

        Note: In the typical single-device-per-process setup, the snapshot only
        contains segments for that process's device, so no device filtering is needed.
        """
        # Clean up the existing comm to start fresh
        self.torchcomm.finalize()
        self.torchcomm = None
        self.wrapper = None

        # Allocate memory BEFORE creating the comm
        # This allocation will be in CUDACachingAllocator's snapshot when
        # registerMemPreHook runs (triggered by CCA hook attachment)
        device = self.get_device()
        count = 110 * 1024 * 1024  # Large allocation spanning multiple segments
        dtype = torch.uint8

        rank, num_ranks = get_rank_and_size()

        # Pre-allocate memory - this is the key: memory exists BEFORE the CCA hook
        pre_allocated_tensor = torch.ones(count, dtype=dtype, device=device) * float(
            rank + 1
        )

        # Now create a new comm - this attaches the CCA hook which triggers
        # registerMemPreHook() to:
        # 1. Get snapshot from CUDACachingAllocator
        # 2. Find our pre-allocated tensor's segments
        # 3. Register them globally via global_register_address()
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()

        # Use the pre-allocated tensor in a collective
        # If registerMemPreHook worked correctly, the memory is already registered
        # and the collective should succeed using ctran
        work = self.torchcomm.all_reduce(pre_allocated_tensor, ReduceOp.SUM, False)
        work.wait()

        # Verify collective result
        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(pre_allocated_tensor.cpu(), float(expected))
        torch.testing.assert_close(pre_allocated_tensor.cpu(), expected_tensor)

        print(
            f"rank {rank}: pre-allocated memory correctly registered and used in collective"
        )


if __name__ == "__main__":
    unittest.main()
