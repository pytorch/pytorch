"""
Inductor PyTorch test runners.

This module contains Python implementations for Inductor-related PyTorch test functions,
replacing shell-based implementations with native Python logic.
"""

import os
import logging
import subprocess
from typing import List, Optional

from .test_execution import PyTorchTestRunner


class InductorDistributedTestRunner(PyTorchTestRunner):
    """Runner for Inductor distributed tests."""
    
    def run_inductor_distributed_tests(self) -> bool:
        """
        Run Inductor distributed tests.
        
        This replaces the test_inductor_distributed shell function.
        """
        self.logger.info("Running Inductor distributed tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Inductor distributed test files
        test_files = [
            "distributed/_composable/test_replicate.py",
            "distributed/test_inductor_collectives.py"
        ]
        
        # Run Inductor distributed tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorShardTestRunner(PyTorchTestRunner):
    """Runner for Inductor shard tests."""
    
    def run_inductor_shard_tests(self, shard_id: str, num_shards: str) -> bool:
        """
        Run Inductor tests for a specific shard.
        
        This replaces the test_inductor_shard shell function.
        """
        self.logger.info(f"Running Inductor shard tests: {shard_id}/{num_shards}")
        
        # Validate shard parameters
        if not num_shards:
            self.logger.error("NUM_TEST_SHARDS must be defined to run Inductor test shard")
            return False
        
        # Get environment variables
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--include", "inductor"])
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


class InductorAOTITestRunner(PyTorchTestRunner):
    """Runner for Inductor AOTI tests."""
    
    def run_inductor_aoti_tests(self) -> bool:
        """
        Run Inductor AOTI (Ahead-of-Time Inference) tests.
        
        This replaces the test_inductor_aoti shell function.
        """
        self.logger.info("Running Inductor AOTI tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Inductor AOTI test files
        test_files = [
            "inductor/test_aoti_package.py",
            "inductor/test_aot_inductor.py"
        ]
        
        # Run Inductor AOTI tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorCppWrapperTestRunner(PyTorchTestRunner):
    """Runner for Inductor C++ wrapper tests."""
    
    def run_inductor_cpp_wrapper_shard_tests(self, shard_id: str, num_shards: str) -> bool:
        """
        Run Inductor C++ wrapper tests for a specific shard.
        
        This replaces the test_inductor_cpp_wrapper_shard shell function.
        """
        self.logger.info(f"Running Inductor C++ wrapper shard tests: {shard_id}/{num_shards}")
        
        # Validate shard parameters
        if not num_shards:
            self.logger.error("NUM_TEST_SHARDS must be defined to run Inductor C++ wrapper test shard")
            return False
        
        # Get environment variables
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set C++ wrapper environment
        env_vars = {
            "TORCHINDUCTOR_CPP_WRAPPER": "1"
        }
        
        # Build command parts
        cmd_parts = ["python", "test/run_test.py"]
        cmd_parts.extend(["--include", "inductor"])
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
            self.logger.error(f"Inductor C++ wrapper shard tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Inductor C++ wrapper shard tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorTritonTestRunner(PyTorchTestRunner):
    """Runner for Inductor Triton tests."""
    
    def run_inductor_triton_cpu_tests(self) -> bool:
        """
        Run Inductor Triton CPU tests.
        
        This replaces the test_inductor_triton_cpu shell function.
        """
        self.logger.info("Running Inductor Triton CPU tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set Triton CPU environment
        env_vars = {
            "TRITON_INTERPRET": "1"
        }
        
        # Inductor Triton CPU test files
        test_files = [
            "inductor/test_triton_kernels.py"
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
            self.logger.error(f"Inductor Triton CPU tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Inductor Triton CPU tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorHalideTestRunner(PyTorchTestRunner):
    """Runner for Inductor Halide tests."""
    
    def run_inductor_halide_tests(self) -> bool:
        """
        Run Inductor Halide tests.
        
        This replaces the test_inductor_halide shell function.
        """
        self.logger.info("Running Inductor Halide tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set Halide environment
        env_vars = {
            "TORCHINDUCTOR_HALIDE": "1"
        }
        
        # Inductor Halide test files
        test_files = [
            "inductor/test_halide.py"
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
            self.logger.error(f"Inductor Halide tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Inductor Halide tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
