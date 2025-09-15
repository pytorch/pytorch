#!/usr/bin/env python3
"""
Test for distributed logging functionality that prevents log spew.

Owner(s): ["distributed"]
"""

import io
import warnings
from contextlib import redirect_stderr

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests


class DistributedLoggingTest(MultiProcessTestCase):
    """Test that logging/warnings only appear on rank 0 in distributed settings."""

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    def test_what_users_see(self):
        """Test showing exactly what users will see with and without the patch."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        # Capture what each rank would emit
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            warnings.warn(f"Test warning from rank {self.rank}")
        
        rank_output = stderr_capture.getvalue()
        
        # Show what each rank produces
        print(f"Rank {self.rank} emits: '{rank_output.strip()}'")
        
        # Verify the behavior
        if self.rank == 0:
            self.assertIn("Test warning", rank_output, "Rank 0 should emit warning")
            print(f"Rank {self.rank}: ✓ Warning emitted (expected)")
        else:
            self.assertEqual(rank_output, "", "Non-rank-0 should emit nothing")
            print(f"Rank {self.rank}: ✓ No output (patch working)")
        
        # Show final result
        if self.rank == 0:
            if rank_output.strip():
                print(f"USER SEES: {rank_output.strip()}")
            else:
                print("USER SEES: (nothing)")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()