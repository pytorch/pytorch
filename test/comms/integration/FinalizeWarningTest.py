#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Test that TorchComm backends log a warning but do not abort when finalize() is
not called before destruction.
"""

import gc
import os
import time
import unittest

from integration.helpers.TorchCommTestHelpers import (
    create_store,
    get_rank_and_size,
    maybe_set_rank_envs,
)

import torch
from torch.comms import new_comm


class FinalizeWarningTest(unittest.TestCase):
    """
    Test class for verifying that TorchComm backends do not abort when finalize
    is not called before destruction.

    This test verifies that:
    1. A TorchComm can be destroyed without calling finalize()
    2. The destructor logs a warning instead of aborting
    3. The process continues to run normally after the comm is destroyed
    """

    def test_destructor_without_finalize_does_not_abort(self):
        """
        Test that destroying a TorchComm without calling finalize() does not abort.

        This test creates a TorchComm, uses it, and then lets it be garbage
        collected without calling finalize(). The test passes if the process
        does not abort and continues running normally.
        """
        maybe_set_rank_envs()

        backend = os.getenv("TEST_BACKEND")
        if backend is None:
            raise RuntimeError("TEST_BACKEND environment variable is not set")

        rank, size = get_rank_and_size()

        # Determine device
        if torch.accelerator.is_available():
            device_count = torch.accelerator.device_count()
            if device_count > 0:
                device_id = rank % device_count
                accelerator = torch.accelerator.current_accelerator()
                assert accelerator is not None
                device_type = accelerator.type
                device = torch.device(f"{device_type}:{device_id}")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        store = create_store()

        # Create a TorchComm
        comm = new_comm(
            backend,
            device,
            store=store,
            name="finalize_warning_test_comm",
        )

        # Use the comm to ensure it is properly initialized
        _ = torch.ones(4, dtype=torch.float, device=device) * float(rank + 1)
        work = comm.barrier(False)
        work.wait()

        print(f"Rank {rank}: Barrier completed, now deleting comm without finalize()")

        # IMPORTANT: Delete the work object BEFORE the comm. TorchWorkNCCL
        # holds a shared_ptr to TorchCommNCCL, so if `work` is alive when
        # `del comm` is called, the comm won't actually be destroyed. We must
        # also wait for the timeout watchdog thread (which runs every 1s) to
        # garbage-collect the work item from its internal queue, breaking the
        # circular reference: comm -> workq -> work_item -> comm.
        del work
        time.sleep(2)

        # Delete the comm without calling finalize()
        # This should log a warning but NOT abort
        del comm

        # Force garbage collection to ensure the destructor runs
        gc.collect()

        # If we reach here, the test passes - the destructor did not abort
        print(f"Rank {rank}: Comm deleted successfully without abort")

        # Perform another operation to verify the process is still healthy
        # Create a new comm to verify the process can still function
        store2 = create_store()
        comm2 = new_comm(
            backend,
            device,
            store=store2,
            name="finalize_warning_test_comm_2",
        )

        # Use the new comm
        work2 = comm2.barrier(False)
        work2.wait()

        print(f"Rank {rank}: Second comm created and used successfully")

        # Clean up properly this time
        comm2.finalize()
        del work2
        del comm2
        gc.collect()

        print(f"Rank {rank}: Test completed successfully")


if __name__ == "__main__":
    unittest.main()
