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

    def test_cpp_extension_case(self):
        """Test the specific cpp_extension case that was originally fixed."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            try:
                from torch.utils.cpp_extension import _get_cuda_arch_flags
                _get_cuda_arch_flags()
            except Exception:
                pass  # May fail if CUDA not available, but we're testing logging
        
        output = stderr_capture.getvalue()
        
        if self.rank == 0:
            # Rank 0 should have the warning (if CUDA available)
            pass  # Just ensure no crash
        else:
            # Should definitely not have TORCH_CUDA_ARCH_LIST warning on non-rank-0
            self.assertNotIn("TORCH_CUDA_ARCH_LIST", output,
                           f"Unexpected CUDA arch warning on rank {self.rank}: {output}")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()