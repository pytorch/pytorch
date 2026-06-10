# -*- coding: utf-8 -*-
import os
import torch
import traceback
import multiprocessing as mp
from typing import Tuple,List
import unittest
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_WINDOWS,
    run_tests,
    TestCase,
)

@unittest.skipIf(not(IS_WINDOWS or IS_LINUX), "Test only support Windows and linux")
class TestUnwindFunctionality(TestCase):
    """Test case for verifying PyTorch unwind functionality with TORCH_SHOW_CPP_STACKTRACES"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store original environment variable state
        self.original_stacktrace_env = os.environ.get("TORCH_SHOW_CPP_STACKTRACES")
        # Set spawn start method for Windows compatibility
        mp.set_start_method('spawn', force=True)

    def setUp(self):
        """Set up test environment (called before each test method)"""
        # Set the environment variable for this test
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    def tearDown(self):
        """Restore original environment state (called after each test method)"""
        if self.original_stacktrace_env is not None:
            os.environ["TORCH_SHOW_CPP_STACKTRACES"] = self.original_stacktrace_env
        else:
            # Remove the variable if it didn't exist originally
            if "TORCH_SHOW_CPP_STACKTRACES" in os.environ:
                del os.environ["TORCH_SHOW_CPP_STACKTRACES"]

    @staticmethod
    def trigger_torch_check_fail(conn) -> None:
        """Child process function to trigger torchCheckFail exception"""
        try:
            # Force CPU device to avoid GPU dependencies
            torch.set_default_device("cpu")
            
            # Create tensors with incompatible shapes for matmul
            tensor1 = torch.randn(2, 3)
            tensor2 = torch.randn(4, 5)
            _ = torch.matmul(tensor1, tensor2)
            
            # Send empty data if no exception
            conn.send(("", ""))
        except RuntimeError:
            # Capture full stack trace including C++ frames
            full_traceback = traceback.format_exc()
            conn.send(("RuntimeError", full_traceback))
        finally:
            conn.close()

    @staticmethod
    def check_cpp_stacktrace(traceback_str: str) -> Tuple[bool, List]:
        """Verify presence of C++ stack frames in traceback"""
        # Key indicators of C++ stack unwinding on Windows/linux
        torch_file_path = "/aten/src/ATen/native/" if IS_LINUX else "\\aten\\src\\ATen\\native\\"
        cpp_stack_keywords = [
            "c10::detail::torchCheckFail",
            "at::native::matmul",
            "at::_ops::mm::call",
            "#",
            torch_file_path
        ]
        
        matched_keywords = []
        for keyword in cpp_stack_keywords:
            if keyword in traceback_str:
                matched_keywords.append(keyword)
        
        # Consider unwind successful if at least 3 keywords match
        is_unwind_working = len(matched_keywords) >= 3
        return is_unwind_working, matched_keywords

    def test_unwind_functionality(self):
        """Main test method for unwind functionality verification"""
        # Create pipe for inter-process communication
        parent_conn, child_conn = mp.Pipe()
        
        # Start child process
        p = mp.Process(
            target=self.trigger_torch_check_fail,
            args=(child_conn,)
        )
        
        try:
            p.start()
            p.join(timeout=30)
        except TimeoutError:
            p.terminate()
            self.fail("Child process timed out after 30 seconds")
        except Exception as e:
            self.fail(f"Failed to start child process: {str(e)}")
        
        # Receive traceback from child process
        if not parent_conn.poll():
            self.fail("No traceback data received from child process")
        
        exc_type, traceback_str = parent_conn.recv()
        
        # Validate exception type
        self.assertEqual(exc_type, "RuntimeError", "Expected RuntimeError not raised")
        
        # Validate traceback content
        self.assertTrue(len(traceback_str) > 0, "Empty traceback received")
        
        # Check for C++ stack frames
        is_unwind_working, matched_keywords = self.check_cpp_stacktrace(traceback_str)
        
        # Assert unwind functionality works
        self.assertTrue(
            is_unwind_working,
            f"Unwind functionality failed. Matched keywords: {matched_keywords}"
        )

if __name__ == "__main__":
    run_tests()