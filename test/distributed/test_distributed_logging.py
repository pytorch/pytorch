#!/usr/bin/env python3
"""
Test for distributed logging functionality that prevents log spew.

Owner(s): ["distributed"]
"""

import functools
import io
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if TEST_WITH_DEV_DBG_ASAN:
    import pytest
    pytestmark = pytest.mark.skip(reason="Skip dev-asan as torch + multiprocessing spawn have known issues")


class DistributedLoggingTest(MultiProcessTestCase):
    """Test that logging/warnings only appear on rank 0 in distributed settings."""

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    def _verify_rank_0_only_output(self, test_func, expected_messages=None):
        """Helper to verify that output only appears on rank 0."""
        rank = self.rank
        
        # Capture all output
        stderr_capture = io.StringIO()
        stdout_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture), redirect_stdout(stdout_capture):
            test_func()
        
        stderr_output = stderr_capture.getvalue()
        stdout_output = stdout_capture.getvalue()
        
        if rank == 0:
            # Rank 0 should have output
            self.assertTrue(
                stderr_output or stdout_output,
                f"Expected output on rank 0, but got none. stderr: '{stderr_output}', stdout: '{stdout_output}'"
            )
            
            if expected_messages:
                combined_output = stderr_output + stdout_output
                for msg in expected_messages:
                    self.assertIn(msg, combined_output, 
                                f"Expected message '{msg}' not found in output")
        else:
            # Non-rank-0 should have no warning/logging output
            # (but may have other output like test framework messages)
            self.assertNotIn("UserWarning:", stderr_output,
                           f"Unexpected warning on rank {rank}: {stderr_output}")
            self.assertNotIn("WARNING:", stderr_output,
                           f"Unexpected logging on rank {rank}: {stderr_output}")

    def test_warnings_only_on_rank_0(self):
        """Test that warnings.warn only emits on rank 0."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        def emit_warnings():
            warnings.warn("Test warning 1")
            warnings.warn("Test warning 2")
            # Duplicate should be deduplicated
            warnings.warn("Test warning 1")
        
        self._verify_rank_0_only_output(emit_warnings, ["Test warning 1", "Test warning 2"])
        
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
        
        def emit_logs():
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            
            # Add a handler to capture output
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            
            logger.warning("Logger warning message")
            logger.info("Logger info message")
            logger.debug("Logger debug message")
            
            # Module-level logging
            logging.warning("Module warning message")
            logging.info("Module info message")
        
        self._verify_rank_0_only_output(
            emit_logs, 
            ["Logger warning", "Module warning"]
        )
        
        dist.destroy_process_group()

    def test_deduplication(self):
        """Test that duplicate warnings are deduplicated."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            # Emit the same warning 5 times
            for _ in range(5):
                warnings.warn("Duplicate warning message")
        
        output = stderr_capture.getvalue()
        
        if self.rank == 0:
            # Should only appear once
            count = output.count("Duplicate warning message")
            self.assertEqual(count, 1, 
                           f"Warning appeared {count} times, expected 1")
        else:
            self.assertEqual(output, "", "Expected no output on non-rank-0")
        
        dist.destroy_process_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_with_nccl_backend(self):
        """Test distributed logging with NCCL backend."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        def emit_warnings():
            warnings.warn("NCCL test warning")
            logging.warning("NCCL test logging")
        
        self._verify_rank_0_only_output(
            emit_warnings,
            ["NCCL test warning", "NCCL test logging"]
        )
        
        dist.destroy_process_group()

    def test_force_log_on_all_ranks(self):
        """Test that force_log_on_all_ranks works."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        from torch.distributed._distributed_logging import force_log_on_all_ranks
        
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            force_log_on_all_ranks("Critical message for all ranks")
        
        output = stdout_capture.getvalue()
        
        # All ranks should have this message
        self.assertIn(f"[Rank {self.rank}]", output)
        self.assertIn("Critical message for all ranks", output)
        
        dist.destroy_process_group()

    def test_distributed_print(self):
        """Test distributed_print utility."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        from torch.distributed._distributed_logging import distributed_print
        
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            distributed_print("Test message", rank_prefix=True)
            distributed_print("Another message", rank_prefix=False)
        
        output = stdout_capture.getvalue()
        
        if self.rank == 0:
            self.assertIn("Test message", output)
            self.assertIn("Another message", output)
            self.assertIn("[Rank 0]", output)  # From rank_prefix=True
        else:
            self.assertEqual(output, "", "Expected no output on non-rank-0")
        
        dist.destroy_process_group()

    def test_cpp_extension_case(self):
        """Test the specific cpp_extension case from the PR."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        # Set debug level to trigger the message
        logging.getLogger('torch.utils.cpp_extension').setLevel(logging.DEBUG)
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            try:
                from torch.utils.cpp_extension import _get_cuda_arch_flags
                _get_cuda_arch_flags()
            except Exception:
                pass  # May fail if CUDA not available, but we're testing logging
        
        output = stderr_capture.getvalue()
        
        if self.rank == 0:
            # May or may not have output depending on CUDA availability
            pass
        else:
            # Should definitely not have TORCH_CUDA_ARCH_LIST warning on non-rank-0
            self.assertNotIn("TORCH_CUDA_ARCH_LIST", output,
                           f"Unexpected CUDA arch warning on rank {self.rank}")
        
        dist.destroy_process_group()

    def test_unpatch_restore(self):
        """Test that unpatching restores original behavior."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        
        from torch.distributed._distributed_logging import unpatch_logging, patch_logging_for_distributed
        
        # First unpatch to restore original behavior
        unpatch_logging()
        
        stderr_capture = io.StringIO()
        
        with redirect_stderr(stderr_capture):
            warnings.warn("Should appear on all ranks")
        
        output = stderr_capture.getvalue()
        
        # All ranks should have output after unpatching
        self.assertIn("Should appear on all ranks", output,
                     f"Warning should appear on rank {self.rank} after unpatch")
        
        # Re-patch for cleanup
        patch_logging_for_distributed()
        
        dist.destroy_process_group()


instantiate_parametrized_tests(DistributedLoggingTest)

if __name__ == "__main__":
    run_tests()