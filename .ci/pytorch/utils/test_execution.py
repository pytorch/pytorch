"""
Test execution utilities for PyTorch CI.

This module provides utilities for executing PyTorch tests in Python,
replacing shell-based test functions with native Python implementations.
"""

import os
import subprocess
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from .shell_utils import run_command_with_output, get_pytorch_root


class TestExecutionError(Exception):
    """Exception raised when test execution fails."""
    pass


class PyTorchTestRunner:
    """Utility class for running PyTorch tests."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.pytorch_root = get_pytorch_root()
        
    def run_test_suite(
        self,
        test_files: List[str],
        extra_options: Optional[str] = None,
        upload_artifacts: bool = True,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Run a test suite using PyTorch's run_test.py.
        
        Args:
            test_files: List of test files to run (relative to test/ directory)
            extra_options: Extra options to pass to the test runner
            upload_artifacts: Whether to upload artifacts while running
            timeout: Timeout in seconds
            
        Returns:
            True if all tests pass, False otherwise
        """
        success = True
        
        for test_file in test_files:
            if not self._run_single_test(test_file, extra_options, upload_artifacts, timeout):
                success = False
                
        return success
    
    def _run_single_test(
        self,
        test_file: str,
        extra_options: Optional[str] = None,
        upload_artifacts: bool = True,
        timeout: Optional[int] = None
    ) -> bool:
        """Run a single test file."""
        # Build command
        cmd_parts = ["python", "test/run_test.py", "--include", test_file]
        
        # Add extra options
        if extra_options:
            # Split extra options properly, handling quoted strings
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
        
        # Add artifact upload option
        if upload_artifacts:
            cmd_parts.append("--upload-artifacts-while-running")
            
        command = " ".join(cmd_parts)
        
        self.logger.info(f"Running test: {test_file}")
        self.logger.debug(f"Command: {command}")
        
        # Record start time for timing
        start_time = time.time()
        
        # Run the command
        success, stdout, stderr = run_command_with_output(
            command,
            cwd=str(self.pytorch_root),
            timeout=timeout
        )
        
        # Log timing
        elapsed = time.time() - start_time
        self.logger.info(f"Test {test_file} completed in {elapsed:.2f}s")
        
        if not success:
            self.logger.error(f"Test {test_file} failed")
            if stderr:
                self.logger.error(f"STDERR: {stderr}")
        else:
            self.logger.info(f"Test {test_file} passed")
            
        return success
    
    def _run_command_with_timing(self, command: str, timeout: Optional[int] = None) -> bool:
        """
        Run a command with timing and return success status.
        
        Args:
            command: The command to execute
            timeout: Timeout in seconds
            
        Returns:
            True if command succeeded, False otherwise
        """
        import time
        
        self.logger.info(f"Executing: {command}")
        
        # Record start time for timing
        start_time = time.time()
        
        # Run the command
        success, stdout, stderr = run_command_with_output(
            command,
            cwd=str(self.pytorch_root),
            timeout=timeout
        )
        
        # Log timing
        elapsed = time.time() - start_time
        self.logger.info(f"Command completed in {elapsed:.2f}s")
        
        if not success:
            self.logger.error(f"Command failed")
            if stderr:
                self.logger.error(f"STDERR: {stderr}")
        else:
            self.logger.info(f"Command passed")
            
        return success
    
    def assert_git_not_dirty(self) -> bool:
        """
        Assert that the git repository is not dirty after tests.
        
        Returns:
            True if git is clean, False otherwise
        """
        self.logger.info("Checking git status...")
        
        success, stdout, stderr = run_command_with_output(
            "git status --porcelain",
            cwd=str(self.pytorch_root)
        )
        
        if not success:
            self.logger.error("Failed to check git status")
            return False
            
        if stdout.strip():
            self.logger.error("Git repository is dirty after tests:")
            self.logger.error(stdout)
            return False
        else:
            self.logger.info("Git repository is clean")
            return True


class DistributedTestRunner(PyTorchTestRunner):
    """Specialized test runner for distributed tests."""
    
    def run_h100_symm_mem_tests(self) -> bool:
        """
        Run H100 symmetric memory tests.
        
        This replaces the test_h100_symm_mem shell function.
        """
        self.logger.info("Running H100 symmetric memory tests")
        
        test_files = [
            "distributed/test_symmetric_memory.py",
            "distributed/test_nvshmem.py", 
            "distributed/test_nvshmem_triton.py",
            "distributed/test_nccl.py"
        ]
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run the test suite
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_h100_distributed_tests(self) -> bool:
        """
        Run H100 distributed tests.
        
        This replaces the test_h100_distributed shell function.
        """
        self.logger.info("Running H100 distributed tests")
        
        test_files = [
            "distributed/test_c10d_nccl.py",
            "distributed/test_c10d_common.py",
            "distributed/test_c10d_spawn.py"
        ]
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run the test suite
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorTestRunner(PyTorchTestRunner):
    """Specialized test runner for Inductor tests."""
    
    def run_h100_cutlass_backend_tests(self) -> bool:
        """
        Run H100 CUTLASS backend tests.
        
        This replaces the test_h100_cutlass_backend shell function.
        """
        self.logger.info("Running H100 CUTLASS backend tests")
        
        # Set CUTLASS environment variable
        cutlass_dir = str(self.pytorch_root / "third_party" / "cutlass")
        env_vars = {"TORCHINDUCTOR_CUTLASS_DIR": cutlass_dir}
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        test_configs = [
            {
                "file": "inductor/test_cutlass_backend",
                "filter": "not addmm"  # -k "not addmm"
            },
            {
                "file": "inductor/test_cutlass_evt",
                "filter": None
            }
        ]
        
        success = True
        
        for config in test_configs:
            # Build command with environment variables
            cmd_parts = ["python", "test/run_test.py", "--include", config["file"]]
            
            # Add filter if specified
            if config["filter"]:
                cmd_parts.extend(["-k", f'"{config["filter"]}"'])
                
            # Add extra options
            if extra_options:
                import shlex
                cmd_parts.extend(shlex.split(extra_options))
                
            cmd_parts.append("--upload-artifacts-while-running")
            
            # Set environment
            test_env = os.environ.copy()
            test_env.update(env_vars)
            
            command = " ".join(cmd_parts)
            self.logger.info(f"Running CUTLASS test: {config['file']}")
            self.logger.debug(f"Command: {command}")
            self.logger.debug(f"Environment: {env_vars}")
            
            start_time = time.time()
            
            # Run with custom environment
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.pytorch_root),
                env=test_env,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"CUTLASS test {config['file']} completed in {elapsed:.2f}s")
            
            if result.returncode != 0:
                self.logger.error(f"CUTLASS test {config['file']} failed")
                if result.stderr:
                    self.logger.error(f"STDERR: {result.stderr}")
                success = False
            else:
                self.logger.info(f"CUTLASS test {config['file']} passed")
                
        return success


class SmokeTestRunner(PyTorchTestRunner):
    """Specialized test runner for smoke tests."""
    
    def run_python_smoke_tests(self) -> bool:
        """
        Run Python smoke tests.
        
        This replaces the test_python_smoke shell function.
        """
        self.logger.info("Running Python smoke tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Run basic smoke test
        success = self.run_test_suite(
            test_files=["test_torch.py"],
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        return success
