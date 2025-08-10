"""
Specialized PyTorch test runners.

This module contains Python implementations for specialized PyTorch test functions,
replacing shell-based implementations with native Python logic.
"""

import os
import logging
import subprocess
from typing import List, Optional

from .test_execution import PyTorchTestRunner


class XLATestRunner(PyTorchTestRunner):
    """Runner for XLA tests."""
    
    def run_xla_tests(self) -> bool:
        """
        Run XLA tests.
        
        This replaces the test_xla shell function.
        """
        self.logger.info("Running XLA tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # XLA test files
        test_files = [
            "test_xla.py"
        ]
        
        # Run XLA tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class ForwardBackwardCompatibilityTestRunner(PyTorchTestRunner):
    """Runner for forward/backward compatibility tests."""
    
    def run_forward_backward_compatibility_tests(self) -> bool:
        """
        Run forward/backward compatibility tests.
        
        This replaces the test_forward_backward_compatibility shell function.
        """
        self.logger.info("Running forward/backward compatibility tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Forward/backward compatibility test files
        test_files = [
            "test_serialization.py",
            "test_backward_compatibility.py"
        ]
        
        # Run forward/backward compatibility tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class BazelTestRunner(PyTorchTestRunner):
    """Runner for Bazel tests."""
    
    def run_bazel_tests(self) -> bool:
        """
        Run Bazel tests.
        
        This replaces the test_bazel shell function.
        """
        self.logger.info("Running Bazel tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Bazel test files
        test_files = [
            "test_bazel.py"
        ]
        
        # Run Bazel tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class ExecutorchTestRunner(PyTorchTestRunner):
    """Runner for ExecuTorch tests."""
    
    def run_executorch_tests(self) -> bool:
        """
        Run ExecuTorch tests.
        
        This replaces the test_executorch shell function.
        """
        self.logger.info("Running ExecuTorch tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # ExecuTorch test files
        test_files = [
            "test_executorch.py"
        ]
        
        # Run ExecuTorch tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class LinuxAarch64TestRunner(PyTorchTestRunner):
    """Runner for Linux AArch64 tests."""
    
    def run_linux_aarch64_tests(self) -> bool:
        """
        Run Linux AArch64 tests.
        
        This replaces the test_linux_aarch64 shell function.
        """
        self.logger.info("Running Linux AArch64 tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Linux AArch64 test files
        test_files = [
            "test_aarch64.py"
        ]
        
        # Run Linux AArch64 tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class OperatorBenchmarkTestRunner(PyTorchTestRunner):
    """Runner for operator benchmark tests."""
    
    def run_operator_benchmark_tests(self) -> bool:
        """
        Run operator benchmark tests.
        
        This replaces the test_operator_benchmark shell function.
        """
        self.logger.info("Running operator benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Operator benchmark test files
        test_files = [
            "test_operator_benchmark.py"
        ]
        
        # Run operator benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class WithoutNumpyTestRunner(PyTorchTestRunner):
    """Runner for tests without NumPy."""
    
    def run_without_numpy_tests(self) -> bool:
        """
        Run tests without NumPy.
        
        This replaces the test_without_numpy shell function.
        """
        self.logger.info("Running tests without NumPy")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set environment to exclude NumPy
        env_vars = {
            "USE_NUMPY": "0"
        }
        
        # Tests without NumPy test files
        test_files = [
            "test_torch.py"
        ]
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--include"] + test_files)
        cmd_parts.append("--verbose")
        
        # Add extra options
        if extra_options:
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
            
        cmd_parts.append("--upload-artifacts-while-running")
        
        # Set environment
        test_env = os.environ.copy()
        test_env.update(env_vars)
        
        command = " ".join(cmd_parts)
        self.logger.info(f"Executing: {command}")
        self.logger.debug(f"Environment: {env_vars}")
        
        # Run with custom environment
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.pytorch_root),
            env=test_env,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        
        if not success:
            self.logger.error(f"Tests without NumPy failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Tests without NumPy passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class LazyTensorMetaReferenceTestRunner(PyTorchTestRunner):
    """Runner for lazy tensor meta reference tests."""
    
    def run_lazy_tensor_meta_reference_disabled_tests(self) -> bool:
        """
        Run lazy tensor meta reference disabled tests.
        
        This replaces the test_lazy_tensor_meta_reference_disabled shell function.
        """
        self.logger.info("Running lazy tensor meta reference disabled tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set environment to disable meta reference
        env_vars = {
            "TORCH_DISABLE_FUNCTIONALIZATION_META_REFERENCE": "1"
        }
        
        # Lazy tensor meta reference test files
        test_files = [
            "test_lazy.py"
        ]
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--include"] + test_files)
        cmd_parts.append("--verbose")
        
        # Add extra options
        if extra_options:
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
            
        cmd_parts.append("--upload-artifacts-while-running")
        
        # Set environment
        test_env = os.environ.copy()
        test_env.update(env_vars)
        
        command = " ".join(cmd_parts)
        self.logger.info(f"Executing: {command}")
        self.logger.debug(f"Environment: {env_vars}")
        
        # Run with custom environment
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.pytorch_root),
            env=test_env,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        
        if not success:
            self.logger.error(f"Lazy tensor meta reference disabled tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Lazy tensor meta reference disabled tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class DynamoWrappedShardTestRunner(PyTorchTestRunner):
    """Runner for Dynamo wrapped shard tests."""
    
    def run_dynamo_wrapped_shard_tests(self, shard_id: str, num_shards: str) -> bool:
        """
        Run Dynamo wrapped shard tests.
        
        This replaces the test_dynamo_wrapped_shard shell function.
        """
        self.logger.info(f"Running Dynamo wrapped shard tests: {shard_id}/{num_shards}")
        
        # Validate shard parameters
        if not num_shards:
            self.logger.error("NUM_TEST_SHARDS must be defined to run Dynamo wrapped shard tests")
            return False
        
        # Get environment variables
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set Dynamo environment
        env_vars = {
            "PYTORCH_TEST_WITH_DYNAMO": "1"
        }
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--include", "test_modules.py"])
        cmd_parts.extend(["--shard", shard_id, num_shards])
        cmd_parts.append("--verbose")
        
        # Add extra options
        if extra_options:
            import shlex
            cmd_parts.extend(shlex.split(extra_options))
            
        cmd_parts.append("--upload-artifacts-while-running")
        
        # Set environment
        test_env = os.environ.copy()
        test_env.update(env_vars)
        
        command = " ".join(cmd_parts)
        self.logger.info(f"Executing: {command}")
        self.logger.debug(f"Environment: {env_vars}")
        
        # Run with custom environment
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.pytorch_root),
            env=test_env,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        
        if not success:
            self.logger.error(f"Dynamo wrapped shard tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Dynamo wrapped shard tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class EinopsTestRunner(PyTorchTestRunner):
    """Runner for Einops tests."""
    
    def run_einops_tests(self) -> bool:
        """
        Run Einops tests.
        
        This replaces the test_einops shell function.
        """
        self.logger.info("Running Einops tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Einops test files
        test_files = [
            "test_einops.py"
        ]
        
        # Run Einops tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
