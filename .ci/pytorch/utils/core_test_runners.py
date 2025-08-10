"""
Core PyTorch test runners.

This module contains Python implementations for core PyTorch test functions,
replacing shell-based implementations with native Python logic.
"""

import os
import logging
from typing import List, Optional

from .test_execution import PyTorchTestRunner


class PythonTestRunner(PyTorchTestRunner):
    """Runner for core Python tests."""
    
    def run_python_tests(self) -> bool:
        """
        Run core Python tests.
        
        This replaces the test_python shell function.
        """
        self.logger.info("Running core Python tests")
        
        # Get environment variables
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        include_clause = os.environ.get("INCLUDE_CLAUSE", "")
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--exclude-jit-executor", "--exclude-distributed-tests"])
        
        # Add include clause if specified
        if include_clause:
            import shlex
            cmd_parts.extend(shlex.split(include_clause))
        
        cmd_parts.append("--verbose")
        
        # Add extra options
        if extra_options:
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
        
        command = " ".join(cmd_parts)
        
        self.logger.info(f"Executing: {command}")
        
        # Run the test
        success = self._run_command_with_timing(command)
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_python_shard_tests(self, shard_id: str, num_shards: str) -> bool:
        """
        Run Python tests for a specific shard.
        
        This replaces the test_python_shard shell function.
        """
        self.logger.info(f"Running Python shard tests: {shard_id}/{num_shards}")
        
        # Validate shard parameters
        if not num_shards:
            self.logger.error("NUM_TEST_SHARDS must be defined to run a Python test shard")
            return False
        
        # Get environment variables
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        include_clause = os.environ.get("INCLUDE_CLAUSE", "")
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--exclude-jit-executor", "--exclude-distributed-tests"])
        
        # Add include clause if specified
        if include_clause:
            import shlex
            cmd_parts.extend(shlex.split(include_clause))
        
        # Add shard parameters
        cmd_parts.extend(["--shard", shard_id, num_shards])
        cmd_parts.append("--verbose")
        
        # Add extra options
        if extra_options:
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
            
        cmd_parts.append("--upload-artifacts-while-running")
        
        command = " ".join(cmd_parts)
        
        self.logger.info(f"Executing: {command}")
        
        # Run the test
        success = self._run_command_with_timing(command)
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_python_legacy_jit_tests(self) -> bool:
        """
        Run Python legacy JIT tests.
        
        This replaces the test_python_legacy_jit shell function.
        """
        self.logger.info("Running Python legacy JIT tests")
        
        # Run legacy JIT tests
        success = self.run_test_suite(
            test_files=["test_jit_legacy", "test_jit_fuser_legacy"],
            extra_options="--verbose",
            upload_artifacts=False
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class AtenTestRunner(PyTorchTestRunner):
    """Runner for ATen tests."""
    
    def run_aten_tests(self) -> bool:
        """
        Run ATen tests.
        
        This replaces the test_aten shell function.
        """
        self.logger.info("Running ATen tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run ATen tests
        success = self.run_test_suite(
            test_files=["test_ops.py", "test_ops_gradients.py", "test_ops_fwd_gradients.py"],
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class Vec256TestRunner(PyTorchTestRunner):
    """Runner for Vec256 tests."""
    
    def run_vec256_tests(self) -> bool:
        """
        Run Vec256 tests.
        
        This replaces the test_vec256 shell function.
        """
        self.logger.info("Running Vec256 tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run Vec256 tests
        success = self.run_test_suite(
            test_files=["test_cpp_extensions_aot_no_ninja.py", "test_cpp_extensions_aot_ninja.py"],
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class DocsTestRunner(PyTorchTestRunner):
    """Runner for documentation tests."""
    
    def run_docs_tests(self) -> bool:
        """
        Run documentation tests.
        
        This replaces the test_docs_test shell function.
        """
        self.logger.info("Running documentation tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run documentation tests
        success = self.run_test_suite(
            test_files=["test_docs.py"],
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
