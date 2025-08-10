"""
Benchmark PyTorch test runners.

This module contains Python implementations for benchmark-related PyTorch test functions,
replacing shell-based implementations with native Python logic.
"""

import os
import logging
import subprocess
from typing import List, Optional

from .test_execution import PyTorchTestRunner


class BenchmarkTestRunner(PyTorchTestRunner):
    """Runner for general benchmark tests."""
    
    def run_benchmark_tests(self) -> bool:
        """
        Run general benchmark tests.
        
        This replaces the test_benchmarks shell function.
        """
        self.logger.info("Running benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Benchmark test files
        test_files = [
            "benchmarks/test_benchmark_utils.py"
        ]
        
        # Run benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class DynamoBenchmarkTestRunner(PyTorchTestRunner):
    """Runner for Dynamo benchmark tests."""
    
    def run_dynamo_benchmark_tests(self) -> bool:
        """
        Run Dynamo benchmark tests.
        
        This replaces the test_dynamo_benchmark shell function.
        """
        self.logger.info("Running Dynamo benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Dynamo benchmark test files
        test_files = [
            "dynamo/test_dynamo_utils.py",
            "dynamo/test_compile.py"
        ]
        
        # Run Dynamo benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_single_dynamo_benchmark_tests(self) -> bool:
        """
        Run single Dynamo benchmark tests.
        
        This replaces the test_single_dynamo_benchmark shell function.
        """
        self.logger.info("Running single Dynamo benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Single Dynamo benchmark test files
        test_files = [
            "dynamo/test_compile.py"
        ]
        
        # Run single Dynamo benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class InductorBenchmarkTestRunner(PyTorchTestRunner):
    """Runner for Inductor benchmark tests."""
    
    def run_inductor_micro_benchmark_tests(self) -> bool:
        """
        Run Inductor micro benchmark tests.
        
        This replaces the test_inductor_micro_benchmark shell function.
        """
        self.logger.info("Running Inductor micro benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Inductor micro benchmark test files
        test_files = [
            "inductor/test_micro_benchmarks.py"
        ]
        
        # Run Inductor micro benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_inductor_torchbench_smoketest_perf_tests(self) -> bool:
        """
        Run Inductor TorchBench smoketest performance tests.
        
        This replaces the test_inductor_torchbench_smoketest_perf shell function.
        """
        self.logger.info("Running Inductor TorchBench smoketest performance tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Set TorchBench environment
        env_vars = {
            "PYTORCH_TEST_WITH_SLOW": "1"
        }
        
        # TorchBench smoketest performance test files
        test_files = [
            "inductor/test_torchbench_perf.py"
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
            self.logger.error(f"Inductor TorchBench smoketest performance tests failed")
            if result.stderr:
                self.logger.error(f"STDERR: {result.stderr}")
        else:
            self.logger.info(f"Inductor TorchBench smoketest performance tests passed")
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class CacheBenchTestRunner(PyTorchTestRunner):
    """Runner for cache benchmark tests."""
    
    def run_cachebench_tests(self) -> bool:
        """
        Run cache benchmark tests.
        
        This replaces the test_cachebench shell function.
        """
        self.logger.info("Running cache benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Cache benchmark test files
        test_files = [
            "test_fx_experimental.py"
        ]
        
        # Run cache benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
    
    def run_verify_cachebench_tests(self) -> bool:
        """
        Run verify cache benchmark tests.
        
        This replaces the test_verify_cachebench shell function.
        """
        self.logger.info("Running verify cache benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Verify cache benchmark test files
        test_files = [
            "test_fx_experimental.py"
        ]
        
        # Run verify cache benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class PerfDashboardTestRunner(PyTorchTestRunner):
    """Runner for performance dashboard tests."""
    
    def run_perf_for_dashboard_tests(self) -> bool:
        """
        Run performance tests for dashboard.
        
        This replaces the test_perf_for_dashboard shell function.
        """
        self.logger.info("Running performance tests for dashboard")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Performance dashboard test files
        test_files = [
            "test_profiler.py"
        ]
        
        # Run performance dashboard tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class TorchFunctionBenchmarkTestRunner(PyTorchTestRunner):
    """Runner for torch function benchmark tests."""
    
    def run_torch_function_benchmark_tests(self) -> bool:
        """
        Run torch function benchmark tests.
        
        This replaces the test_torch_function_benchmark shell function.
        """
        self.logger.info("Running torch function benchmark tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Torch function benchmark test files
        test_files = [
            "test_torch_function.py"
        ]
        
        # Run torch function benchmark tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
