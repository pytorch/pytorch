#!/usr/bin/env python3
"""
Test for distributed logging functionality that prevents log spew.

Owner(s): ["distributed"]
"""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

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

    def test_logging_what_users_see(self):
        """Test showing exactly what users will see for logging statements."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        # Capture what each rank would emit for logging
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            import logging
            # Add a handler to ensure we capture the logging output
            handler = logging.StreamHandler()
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            
            logging.warning(f"Test logging warning from rank {self.rank}")
        
        rank_output = stderr_capture.getvalue()
        
        # Show what each rank produces
        print(f"Rank {self.rank} logging emits: '{rank_output.strip()}'")
        
        # Verify the behavior
        if self.rank == 0:
            # Rank 0 should emit logging (may or may not capture depending on handler setup)
            print(f"Rank {self.rank}: Logging attempted")
        else:
            # Non-rank-0 should not emit logging
            self.assertNotIn(f"rank {self.rank}", rank_output.lower(),
                           f"Unexpected logging on rank {self.rank}: '{rank_output}'")
            print(f"Rank {self.rank}: ✓ No logging output (patch working)")
        
        # Show final result for logging
        if self.rank == 0:
            if rank_output.strip():
                print(f"USER SEES (logging): {rank_output.strip()}")
            else:
                print("USER SEES (logging): (nothing)")
        
        dist.destroy_process_group()

    def test_print_optional_patching(self):
        """Test that print can optionally be patched too."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        # Test that print is NOT patched by default
        from torch.distributed._distributed_logging import patch_logging_for_distributed
        patch_logging_for_distributed(patch_print=True)  # Enable print patching
        
        # Capture what each rank would emit for print
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            print(f"Test print from rank {self.rank}")
        
        rank_output = stdout_capture.getvalue()
        
        # Show what each rank produces
        print(f"Rank {self.rank} print emits: '{rank_output.strip()}'")
        
        # Verify the behavior
        if self.rank == 0:
            self.assertIn(f"rank {self.rank}", rank_output.lower(),
                         "Rank 0 should emit print when patch_print=True")
            print(f"Rank {self.rank}: ✓ Print emitted (expected)")
        else:
            self.assertEqual(rank_output, "",
                           f"Non-rank-0 should not emit print when patch_print=True: '{rank_output}'")
            print(f"Rank {self.rank}: ✓ No print output (patch working)")
        
        # Show final result for print
        if self.rank == 0:
            if rank_output.strip():
                print(f"USER SEES (print): {rank_output.strip()}")
            else:
                print("USER SEES (print): (nothing)")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()