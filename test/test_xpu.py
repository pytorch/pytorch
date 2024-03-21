# Owner(s): ["module: intel"]

import sys
import unittest

import torch
import torch.xpu._gpu_trace as gpu_trace
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one
_xpu_computation_op_list = [
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "view_as_real",
    "view_as_complex",
    "view",
    "resize_",
    "resize_as_",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
]
_xpu_tensor_factory_op_list = [
    "as_strided",
    "empty",
    "empty_strided",
]
_xpu_not_test_dtype_op_list = [
    "resize_",  # Skipped by CPU
    "resize_as_",  # Skipped by CPU
    "abs",  # Not aligned dtype
]
_xpu_all_op_list = _xpu_computation_op_list + _xpu_tensor_factory_op_list
_xpu_all_ops = [op for op in ops_and_refs if op.name in _xpu_all_op_list]
_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]


class TestXpu(TestCase):
    def test_device_behavior(self):
        current_device = torch.xpu.current_device()
        torch.xpu.set_device(current_device)
        self.assertEqual(current_device, torch.xpu.current_device())

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_behavior(self):
        current_device = torch.xpu.current_device()
        target_device = (current_device + 1) % torch.xpu.device_count()

        with torch.xpu.device(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

        with torch.xpu._DeviceGuard(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

    def test_get_device_properties(self):
        current_device = torch.xpu.current_device()
        device_properties = torch.xpu.get_device_properties(current_device)
        self.assertEqual(device_properties, torch.xpu.get_device_properties(None))
        self.assertEqual(device_properties, torch.xpu.get_device_properties())

        device_name = torch.xpu.get_device_name(current_device)
        self.assertEqual(device_name, torch.xpu.get_device_name(None))
        self.assertEqual(device_name, torch.xpu.get_device_name())

        device_capability = torch.xpu.get_device_capability(current_device)
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)

    def test_wrong_xpu_fork(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.xpu.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        torch.xpu.set_device(0)
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")

    def test_streams(self):
        s0 = torch.xpu.Stream()
        torch.xpu.set_stream(s0)
        s1 = torch.xpu.current_stream()
        self.assertEqual(s0, s1)
        s2 = torch.xpu.Stream()
        self.assertFalse(s0 == s2)
        torch.xpu.set_stream(s2)
        with torch.xpu.stream(s0):
            self.assertEqual(s0, torch.xpu.current_stream())
        self.assertEqual(s2, torch.xpu.current_stream())

    def test_stream_priority(self):
        low, high = torch.xpu.Stream.priority_range()
        s0 = torch.xpu.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device("xpu:0"), s0.device)

        s1 = torch.xpu.Stream(device=0, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device("xpu:0"), s1.device)

    def test_stream_event_repr(self):
        s = torch.xpu.current_stream()
        self.assertTrue("torch.xpu.Stream" in str(s))
        e = torch.xpu.Event()
        self.assertTrue("torch.xpu.Event(uninitialized)" in str(e))
        s.record_event(e)
        self.assertTrue("torch.xpu.Event" in str(e))

    def test_events(self):
        stream = torch.xpu.current_stream()
        event = torch.xpu.Event()
        self.assertTrue(event.query())
        stream.record_event(event)
        event.synchronize()
        self.assertTrue(event.query())

    def test_generator(self):
        torch.manual_seed(2024)
        g_state0 = torch.xpu.get_rng_state()
        torch.manual_seed(1234)
        g_state1 = torch.xpu.get_rng_state()
        self.assertNotEqual(g_state0, g_state1)

        torch.xpu.manual_seed(2024)
        g_state2 = torch.xpu.get_rng_state()
        self.assertEqual(g_state0, g_state2)

        torch.xpu.set_rng_state(g_state1)
        self.assertEqual(g_state1, torch.xpu.get_rng_state())

        torch.manual_seed(1234)
        torch.xpu.set_rng_state(g_state0)
        self.assertEqual(2024, torch.xpu.initial_seed())

    @onlyXPU
    @suppress_warnings
    @ops(_xpu_computation_ops, dtypes=any_common_cpu_xpu_one)
    def test_compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            xpu_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            xpu_results = sample.output_process_fn_grad(xpu_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            self.assertEqual(xpu_results, cpu_results, atol=1e-4, rtol=1e-4)

    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        # Test boolean values other than 0x00 and 0x01 (gh-54789)
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # Map False -> 0 and True -> Random value in [2, 255]
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)

            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            self.assertEqual(expect, actual)


instantiate_device_type_tests(TestXpu, globals(), only_for="xpu")


class TestXpuTrace(TestCase):
    def setUp(self):
        torch._C._activate_gpu_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self):
        gpu_trace.register_callback_for_event_creation(self.mock)

        event = torch.xpu.Event()
        event.record()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self):
        gpu_trace.register_callback_for_event_deletion(self.mock)

        event = torch.xpu.Event()
        event.record()
        event_id = event._as_parameter_.value
        del event
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self):
        gpu_trace.register_callback_for_event_record(self.mock)

        event = torch.xpu.Event()
        event.record()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.xpu.current_stream().sycl_queue
        )

    def test_event_wait_callback(self):
        gpu_trace.register_callback_for_event_wait(self.mock)

        event = torch.xpu.Event()
        event.record()
        event.wait()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.xpu.current_stream().sycl_queue
        )

    def test_device_synchronization_callback(self):
        gpu_trace.register_callback_for_device_synchronization(self.mock)

        torch.xpu.synchronize()
        self.mock.assert_called()

    def test_stream_synchronization_callback(self):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        stream = torch.xpu.Stream()
        stream.synchronize()
        self.mock.assert_called_once_with(stream.sycl_queue)

    def test_event_synchronization_callback(self):
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        event = torch.xpu.Event()
        event.record()
        event.synchronize()
        self.mock.assert_called_once_with(event._as_parameter_.value)


if __name__ == "__main__":
    run_tests()
