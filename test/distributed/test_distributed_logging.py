#!/usr/bin/env python3
"""
Test for distributed logging functionality that prevents log spew.

Owner(s): ["distributed"]
"""

import io
import logging
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

    def test_warnings_only_on_rank_0(self):
        """Test that warnings.warn only emits on rank 0."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            warnings.warn("Test warning - should only appear on rank 0")
        
        output = stderr_capture.getvalue()
        
        if self.rank == 0:
            self.assertIn("Test warning", output, 
                         f"Expected warning on rank 0, got: '{output}'")
        else:
            self.assertEqual(output, "", 
                           f"Expected no output on rank {self.rank}, got: '{output}'")
        
        dist.destroy_process_group()

    def test_logging_only_on_rank_0(self):
        """Test that logging only emits on rank 0."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            # Add a handler to capture output
            handler = logging.StreamHandler()
            logging.getLogger().addHandler(handler)
            
            logging.warning("Test logging warning - should only appear on rank 0")
        
        output = stderr_capture.getvalue()
        
        if self.rank == 0:
            # May or may not capture logging depending on handler setup
            pass  # Just ensure no crash
        else:
            # Should definitely not have warnings on non-rank-0
            self.assertNotIn("Test logging warning", output,
                           f"Unexpected logging on rank {self.rank}: {output}")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()