# Owner(s): ["module: cuda"]

import functools
import sys
import traceback
import unittest
import unittest.mock
from typing import List

import torch
import torch.utils._cuda_trace as cuda_trace
from torch.testing._internal.common_utils import TestCase, run_tests

# NOTE: Each test needs to be run in a brand new process, to reset the registered hooks
# and make sure the CUDA streams are initialized for each test that uses them.

# We cannot import TEST_CUDA from torch.testing._internal.common_cuda here,
# because if we do that, the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = object  # noqa: F811


class TestCudaTrace(TestCase):
    def setUp(self):
        torch._C._activate_cuda_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self):
        cuda_trace.register_callback_for_cuda_event_creation(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self):
        cuda_trace.register_callback_for_cuda_event_deletion(self.mock)

        event = torch.cuda.Event()
        event.record()
        event_id = event._as_parameter_.value
        del event
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self):
        cuda_trace.register_callback_for_cuda_event_record(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_event_wait_callback(self):
        cuda_trace.register_callback_for_cuda_event_wait(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.wait()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_memory_allocation_callback(self):
        cuda_trace.register_callback_for_cuda_memory_allocation(self.mock)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())

    def test_memory_deallocation_callback(self):
        cuda_trace.register_callback_for_cuda_memory_deallocation(self.mock)

        tensor = torch.empty(3, 8, device="cuda")
        data_ptr = tensor.data_ptr()
        del tensor
        self.mock.assert_called_once_with(data_ptr)

    def test_stream_creation_callback(self):
        cuda_trace.register_callback_for_cuda_stream_creation(self.mock)

        torch.cuda.Stream()
        self.mock.assert_called()

    def test_all_trace_callbacks_called(self):
        other = unittest.mock.MagicMock()
        cuda_trace.register_callback_for_cuda_memory_allocation(self.mock)
        cuda_trace.register_callback_for_cuda_memory_allocation(other)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())
        other.assert_called_once_with(tensor.data_ptr())

    def test_trace_offset(self):
        def cb(tb: List[str], *args, **kwargs):
            tb.extend(traceback.format_stack())

        # If this test stops working because of a change in CUDA trace hooks,
        # please make sure to change the constant itself.
        expected_offset = cuda_trace._CUDA_HOOK_TRACE_OFFSET - 1
        tb: List[str] = []

        cuda_trace.register_callback_for_cuda_memory_allocation(
            functools.partial(cb, tb)
        )
        # Tests if the line calling the allocation is at the expected offset in the
        # stack trace using the fact that the unique variable name will appear in it.
        _long_and_unique_alloc_name_ = torch.ones(10, device="cuda")
        self.assertIn("_long_and_unique_alloc_name", tb[expected_offset])


if __name__ == "__main__":
    run_tests()
