"""
Distributed PyTorch test runners.

This module contains Python implementations for distributed PyTorch test functions,
replacing shell-based implementations with native Python logic.
"""

import os
import logging
from typing import List, Optional

from .test_execution import PyTorchTestRunner


class DistributedTestRunner(PyTorchTestRunner):
    """Runner for distributed tests."""
    
    def run_distributed_tests(self) -> bool:
        """
        Run distributed tests.
        
        This replaces the test_distributed shell function.
        """
        self.logger.info("Running distributed tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Distributed test files
        test_files = [
            "distributed/test_c10d_gloo.py",
            "distributed/test_c10d_nccl.py", 
            "distributed/test_c10d_common.py",
            "distributed/test_c10d_spawn.py",
            "distributed/test_store.py",
            "distributed/test_pg_wrapper.py"
        ]
        
        # Run distributed tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class RPCTestRunner(PyTorchTestRunner):
    """Runner for RPC tests."""
    
    def run_rpc_tests(self) -> bool:
        """
        Run RPC tests.
        
        This replaces the test_rpc shell function.
        """
        self.logger.info("Running RPC tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # RPC test files
        test_files = [
            "distributed/rpc/test_faulty_agent.py",
            "distributed/rpc/test_tensorpipe_agent.py",
            "distributed/rpc/test_share_memory.py"
        ]
        
        # Run RPC tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class CustomBackendTestRunner(PyTorchTestRunner):
    """Runner for custom backend tests."""
    
    def run_custom_backend_tests(self) -> bool:
        """
        Run custom backend tests.
        
        This replaces the test_custom_backend shell function.
        """
        self.logger.info("Running custom backend tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Custom backend test files
        test_files = [
            "test_custom_ops.py"
        ]
        
        # Run custom backend tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success


class CustomScriptOpsTestRunner(PyTorchTestRunner):
    """Runner for custom script ops tests."""
    
    def run_custom_script_ops_tests(self) -> bool:
        """
        Run custom script ops tests.
        
        This replaces the test_custom_script_ops shell function.
        """
        self.logger.info("Running custom script ops tests")
        
        # Get extra options from environment
        extra_options = os.environ.get("PYTHON_TEST_EXTRA_OPTION", "")
        
        # Custom script ops test files
        test_files = [
            "test_custom_ops.py"
        ]
        
        # Run custom script ops tests
        success = self.run_test_suite(
            test_files=test_files,
            extra_options=extra_options,
            upload_artifacts=True
        )
        
        # Check git status
        if success:
            success = self.assert_git_not_dirty()
            
        return success
