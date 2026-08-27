# Owner(s): ["module: cuda"]

import unittest
import unittest.mock

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


# NOTE: Each test needs to be run in a brand new process, to reset the registered hooks
# and make sure the CUDA streams are initialized for each test that uses them.


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaTrace(TestCase):
    hw_classification = HardwareClassification.CUDA
    _do_cuda_memory_leak_check = False
    _do_cuda_non_default_stream = False

    def setUp(self):
        super().setUp()
        torch._C._activate_gpu_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self, device):
        gpu_trace.register_callback_for_event_creation(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self, device):
        gpu_trace.register_callback_for_event_deletion(self.mock)

        event = torch.cuda.Event()
        event.record()
        event_id = event._as_parameter_.value
        del event
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self, device):
        gpu_trace.register_callback_for_event_record(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.current_stream().cuda_stream
        )

    def test_event_wait_callback(self, device):
        gpu_trace.register_callback_for_event_wait(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.wait()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.current_stream().cuda_stream
        )

    def test_memory_allocation_callback(self, device):
        gpu_trace.register_callback_for_memory_allocation(self.mock)

        tensor = torch.empty(10, 4, device=device)
        self.mock.assert_called_once_with(tensor.data_ptr())

    def test_memory_deallocation_callback(self, device):
        gpu_trace.register_callback_for_memory_deallocation(self.mock)

        tensor = torch.empty(3, 8, device=device)
        data_ptr = tensor.data_ptr()
        del tensor
        self.mock.assert_called_once_with(data_ptr)

    def test_stream_creation_callback(self, device):
        gpu_trace.register_callback_for_stream_creation(self.mock)

        # see Note [HIP Lazy Streams]
        if torch.version.hip:
            user_stream = torch.cuda.Stream()
            with torch.cuda.stream(user_stream):
                torch.ones(5, device=device)
        else:
            torch.cuda.Stream()

        self.mock.assert_called()

    def test_device_synchronization_callback(self, device):
        gpu_trace.register_callback_for_device_synchronization(self.mock)

        torch.cuda.synchronize()
        self.mock.assert_called()

    def test_stream_synchronization_callback(self, device):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        stream = torch.cuda.Stream()
        stream.synchronize()
        self.mock.assert_called_once_with(stream.cuda_stream)

    def test_event_synchronization_callback(self, device):
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_memcpy_synchronization(self, device):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        tensor = torch.rand(5, device=device)
        tensor.nonzero()
        self.mock.assert_called_once_with(torch.cuda.current_stream().cuda_stream)

    def test_all_trace_callbacks_called(self, device):
        other = unittest.mock.MagicMock()
        gpu_trace.register_callback_for_memory_allocation(self.mock)
        gpu_trace.register_callback_for_memory_allocation(other)

        tensor = torch.empty(10, 4, device=device)
        self.mock.assert_called_once_with(tensor.data_ptr())
        other.assert_called_once_with(tensor.data_ptr())


instantiate_device_type_tests(TestCudaTrace, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
