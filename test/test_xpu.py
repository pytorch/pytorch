# Owner(s): ["module: intel"]
# ruff: noqa: F841

import collections
import contextlib
import ctypes
import gc
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from copy import deepcopy
from itertools import product

import torch
import torch.xpu._gpu_trace as gpu_trace
from torch.testing import make_tensor
from torch.testing._internal.autocast_test_lists import AutocastTestLists, TestAutocast
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)
from torch.testing._internal.common_utils import (
    find_library_location,
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    run_tests,
    serialTest,
    subtest,
    suppress_warnings,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint_sequential


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


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
class TestXpu(TestCase):
    expandable_segments = False

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
        self.assertTrue(device_capability["device_id"] > 0)
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)
        self.assertTrue(device_capability["local_mem_size"] > 0)
        self.assertEqual(
            device_properties.driver_version, device_capability["driver_version"]
        )
        self.assertEqual(device_properties.has_fp16, device_capability["has_fp16"])
        self.assertEqual(device_properties.has_fp64, device_capability["has_fp64"])
        self.assertEqual(
            device_properties.has_atomic64, device_capability["has_atomic64"]
        )
        self.assertEqual(
            device_properties.has_bfloat16_conversions,
            device_capability["has_bfloat16_conversions"],
        )
        self.assertEqual(
            device_properties.has_subgroup_matrix_multiply_accumulate,
            device_capability["has_subgroup_matrix_multiply_accumulate"],
        )
        self.assertEqual(
            device_properties.has_subgroup_matrix_multiply_accumulate_tensor_float32,
            device_capability["has_subgroup_matrix_multiply_accumulate_tensor_float32"],
        )
        self.assertEqual(
            device_properties.has_subgroup_2d_block_io,
            device_capability["has_subgroup_2d_block_io"],
        )
        if int(torch.version.xpu) >= 20250000:
            self.assertEqual(
                device_properties.architecture,
                device_capability["architecture"],
            )
        self.assertEqual(
            len(str(device_properties.uuid)), 36
        )  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        self.assertEqual(len(device_properties.uuid.bytes), 16)

    def test_get_device_capability(self):
        device_capability = torch.xpu.get_device_capability()
        acc_capability = torch.accelerator.get_device_capability()
        supported_dtypes = acc_capability["supported_dtypes"]
        self.assertIn(torch.bool, supported_dtypes)
        self.assertIn(torch.int, supported_dtypes)
        self.assertIn(torch.float, supported_dtypes)
        if device_capability["has_fp16"]:
            self.assertIn(torch.float16, supported_dtypes)
        if device_capability["has_fp64"]:
            self.assertIn(torch.double, supported_dtypes)
        if torch.xpu.is_bf16_supported(including_emulation=True):
            self.assertIn(torch.bfloat16, supported_dtypes)

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
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

    @unittest.skipIf(
        IS_WINDOWS, "Only for lazy initialization on Linux, not applicable on Windows."
    )
    def test_lazy_init(self):
        """Validate that no XPU calls are made during `import torch` call"""

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        test_script = """\
import torch
from torch.multiprocessing import Process
import copy

def run_model(model, input):
    input_xpu = input.clone().to('xpu')
    model_xpu = copy.deepcopy(model).to('xpu')
    loss_xpu = model_xpu(input_xpu).sum()
    loss = model(input).sum()
    torch.testing.assert_close(loss_xpu.cpu(), loss)

def test_multi_process(model, input):
    p = Process(target=run_model, args=(model, input))
    p.start()
    p.join()
    assert p.exitcode == 0

input = torch.rand(32, 3, 224, 224)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3, stride=2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
)

if __name__ == "__main__":
    test_multi_process(model, input)
    test_multi_process(model, input)
    print(torch.xpu.device_count())
"""
        # XPU have extra lines, so get the last line, refer https://github.com/intel/torch-xpu-ops/issues/2261
        rc = check_output(test_script).splitlines()[-1]
        self.assertEqual(rc, str(torch.xpu.device_count()))

    @unittest.skipIf(
        IS_WINDOWS, "Only for lazy initialization on Linux, not applicable on Windows."
    )
    def test_no_xpu_device_query_on_inductor_import(self):
        """Validate that importing torch._inductor.lowering does not trigger XPU device queries"""

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        test_script = """\
import torch
from unittest.mock import patch

call_count = 0
original_getDeviceCount = torch._C._xpu_getDeviceCount

def counting_getDeviceCount():
    global call_count
    call_count += 1
    return original_getDeviceCount()

with patch.object(torch._C, '_xpu_getDeviceCount', counting_getDeviceCount):
    import torch._inductor.lowering

print(call_count)
print(torch.xpu.is_initialized())
"""
        rc = check_output(test_script).splitlines()
        self.assertEqual(
            rc[0],
            "0",
            "Importing torch._inductor.lowering should not query XPU device count",
        )
        self.assertEqual(
            rc[1],
            "False",
            "Importing torch._inductor.lowering should not initialize XPU",
        )

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
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        stream.record_event(start_event)
        time.sleep(0.1)
        stream.record_event(end_event)
        torch.xpu.synchronize()
        if int(torch.version.xpu) >= 20250000:
            self.assertGreater(start_event.elapsed_time(end_event), 0)
        else:
            with self.assertRaisesRegex(
                NotImplementedError,
                "elapsed_time of XPUEvent requires PyTorch to be built with SYCL compiler version 2025.0.0 or newer.",
            ):
                start_event.elapsed_time(end_event)

        event = torch.xpu.Event(enable_timing=True)
        self.assertEqual(event.sycl_event, 0)
        self.assertEqual(event.event_id, 0)

        event.record()
        self.assertNotEqual(event.sycl_event, 0)
        self.assertNotEqual(event.event_id, 0)
        self.assertEqual(event.sycl_event, event.event_id)

    def test_generic_stream_event(self):
        stream = torch.Stream("xpu")
        self.assertEqual(stream.device_index, torch.xpu.current_device())
        xpu_stream = torch.xpu.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertIsInstance(xpu_stream, torch.Stream)
        self.assertTrue(issubclass(type(xpu_stream), torch.Stream))
        self.assertTrue(torch.Stream in type(xpu_stream).mro())
        self.assertEqual(stream.stream_id, xpu_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.xpu.current_stream().stream_id)

        event1 = torch.Event("xpu", enable_timing=True)
        event2 = torch.Event("xpu", enable_timing=True)
        self.assertEqual(event1.event_id, 0)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.xpu.stream(xpu_stream):
            a_xpu = a.to("xpu", non_blocking=True)
            b_xpu = b.to("xpu", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.xpu.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_xpu = a_xpu + b_xpu
        # Here intendionly records another stream.
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_xpu.cpu(), a + b)
        if int(torch.version.xpu) >= 20250000:
            self.assertGreater(event1.elapsed_time(event2), 0)
        else:
            with self.assertRaisesRegex(
                NotImplementedError,
                "elapsedTime requires PyTorch to be built with SYCL compiler version 2025.0.0 or newer.",
            ):
                event1.elapsed_time(event2)
        xpu_event = torch.xpu.Event()
        self.assertIsInstance(xpu_event, torch.Event)
        self.assertTrue(issubclass(type(xpu_event), torch.Event))
        self.assertTrue(torch.Event in type(xpu_event).mro())

    def test_stream_event_compatibility(self):
        s1 = torch.xpu.Stream()
        s2 = torch.xpu.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s1.stream_id)
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s1.sycl_queue
        )
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s1.native_handle
        )
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s2.stream_id)
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s2.sycl_queue
        )
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s2.native_handle
        )
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.current_stream(torch.accelerator.device_count())
        e1 = torch.xpu.Event(enable_timing=True)
        e2 = torch.xpu.Event(enable_timing=True)
        e3 = torch.Event(enable_timing=True)
        s3 = torch.Stream()
        with s3:
            self.assertEqual(torch.accelerator.current_stream(), s3)
            e1.record(s3)
            a = torch.randn(1000)
            a_xpu = a.to("xpu")
            del a_xpu
            e2.record(s3)
            e2.synchronize()
            self.assertGreater(e1.elapsed_time(e2), 0)
            e3.record(s3)
        with self.assertRaisesRegex(
            RuntimeError, "expected other to be a torch.xpu.Event object"
        ):
            e1.elapsed_time(e3)
        with self.assertRaisesRegex(
            RuntimeError, "expected other to be a torch.Event object"
        ):
            e3.elapsed_time(e1)
        with self.assertRaisesRegex(
            RuntimeError, "expected event to be a torch.Event object"
        ):
            s3.record_event(e1)
        with self.assertRaisesRegex(
            RuntimeError,
            "expected stream to be a torch.Stream or torch.xpu.Stream object",
        ):
            e2.record(e2)
        with self.assertRaisesRegex(
            RuntimeError,
            "expected stream to be a torch.Stream or torch.xpu.Stream object",
        ):
            e2.wait(e2)

    def test_device_context_manager(self):
        prev_device = torch.xpu.current_device()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.xpu.current_device(), prev_device)
        self.assertEqual(torch.xpu.current_device(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.xpu.current_device(), 0)
        self.assertEqual(torch.xpu.current_device(), prev_device)

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.xpu.set_device(src_device)
        with torch.accelerator.device_index(dst_device):
            self.assertEqual(torch.xpu.current_device(), 1)
        self.assertEqual(torch.xpu.current_device(), src_device)

    def test_stream_context_manager(self):
        prev_stream = torch.xpu.current_stream()
        with torch.xpu.Stream() as stream:
            self.assertEqual(stream, torch.xpu.current_stream())
        self.assertEqual(prev_stream, torch.xpu.current_stream())

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.xpu.set_device(src_device)
        src_prev_stream = torch.xpu.current_stream(src_device)
        dst_prev_stream = torch.xpu.current_stream(dst_device)
        with torch.xpu.Stream(dst_device) as dst_stream:
            self.assertEqual(dst_device, torch.xpu.current_device())
            self.assertEqual(dst_stream, torch.xpu.current_stream())
            self.assertEqual(src_prev_stream, torch.xpu.current_stream(src_device))
        self.assertEqual(src_device, torch.xpu.current_device())
        self.assertEqual(src_prev_stream, torch.xpu.current_stream())
        self.assertEqual(dst_prev_stream, torch.xpu.current_stream(dst_device))

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

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).xpu()
        y = torch.zeros(2, 5, dtype=torch.int, device="xpu")
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertEqual(q_copy[0].dtype, torch.float)
        self.assertEqual(q_copy[1].dtype, torch.int)
        self.assertEqual(q_copy[2].dtype, torch.float)
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        q_copy[1].fill_(10)
        y.fill_(10)
        self.assertEqual(q_copy[3], y.storage())

    def test_serialization_array_with_empty(self):
        x = [
            torch.randn(4, 4).xpu(),
            torch.tensor([], dtype=torch.float, device=torch.device("xpu")),
        ]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    def test_out_of_memory(self):
        if self.expandable_segments:
            self.skipTest("Skipping OOM test for expandable segments allocator.")
        tensor = torch.zeros(1024, device="xpu")  # noqa: F841

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 800000000.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="xpu")

        with self.assertRaisesRegex(RuntimeError, "XPU out of memory."):
            torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="xpu")

    def test_raises_oom(self):
        if self.expandable_segments:
            self.skipTest("Skipping OOM test for expandable segments allocator.")
        torch.xpu.memory.empty_cache()
        with self.assertRaises(torch.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="xpu")

    @serialTest()
    def test_1mb_allocation_uses_small_block(self):
        gc.collect()
        torch.xpu.empty_cache()
        prev_allocated = torch.xpu.memory_allocated()
        prev_reserved = torch.xpu.memory_reserved()

        # Allocate a 1MB float32 tensor
        one_mb = 1024 * 1024
        a = torch.ones(one_mb // 4, device="xpu")

        b = a.clone()
        for _ in range(5):
            b = b.clone() + 1

        torch.xpu.synchronize()
        current_allocated = torch.xpu.memory_allocated()
        current_reserved = torch.xpu.memory_reserved()
        # Two live tensors remain (a and b), each 1MB
        self.assertEqual(current_allocated, prev_allocated + 2 * 1024 * 1024)
        self.assertEqual(current_reserved, prev_reserved + 4 * 1024 * 1024)  # 4MB

    @serialTest()
    def test_set_per_process_memory_fraction(self):
        gc.collect()
        torch.xpu.empty_cache()
        total_memory = torch.xpu.get_device_properties().total_memory
        fraction = 0.5
        orig_fraction = torch.xpu.get_per_process_memory_fraction()
        with self.assertRaisesRegex(ValueError, "invalid fraction:"):
            torch.xpu.set_per_process_memory_fraction(-0.1)
        with self.assertRaisesRegex(ValueError, "invalid fraction:"):
            torch.xpu.set_per_process_memory_fraction(1.1)

        torch.xpu.set_per_process_memory_fraction(fraction)
        allowed_memory = int(total_memory * 0.49)
        reserved_memory = torch.xpu.memory_reserved()
        application_memory = allowed_memory - reserved_memory
        tensor = torch.empty(application_memory, dtype=torch.int8, device="xpu")
        del tensor
        gc.collect()
        torch.xpu.empty_cache()

        self.assertEqual(fraction, torch.xpu.get_per_process_memory_fraction())

        application_memory = int(total_memory * 0.51)
        with self.assertRaises(torch.OutOfMemoryError):
            _ = torch.empty(application_memory, dtype=torch.int8, device="xpu")

        torch.xpu.set_per_process_memory_fraction(orig_fraction)

    def test_memory_allocation(self):
        torch.xpu.empty_cache()
        prev_allocated = torch.xpu.memory_allocated()
        prev_reserved = torch.xpu.memory_reserved()
        self.assertGreaterEqual(prev_allocated, 0)
        self.assertGreaterEqual(prev_reserved, 0)
        a = torch.ones(10, device="xpu")
        self.assertGreater(torch.xpu.memory_allocated(), prev_allocated)
        self.assertGreaterEqual(torch.xpu.memory_reserved(), prev_reserved)
        del a
        self.assertEqual(torch.xpu.memory_allocated(), prev_allocated)
        torch.xpu.empty_cache()
        self.assertLessEqual(torch.xpu.memory_reserved(), prev_reserved)
        torch.xpu.reset_accumulated_memory_stats()
        # Activate 1kB memory
        prev_active_current = torch.xpu.memory_stats()["active_bytes.all.current"]
        a = torch.randn(256, device="xpu")
        # Detect if the current active memory is 1kB
        self.assertEqual(
            torch.xpu.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(torch.xpu.memory_stats()["active_bytes.all.freed"], 0)
        del a
        self.assertEqual(
            torch.xpu.memory_stats()["active_bytes.all.current"], prev_active_current
        )
        self.assertEqual(torch.xpu.memory_stats()["active_bytes.all.freed"], 1024)

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_device_memory_allocated(self):
        device_count = torch.xpu.device_count()
        current_alloc = [torch.xpu.memory_allocated(idx) for idx in range(device_count)]
        a = torch.ones(10, device="xpu:0")
        self.assertGreater(torch.xpu.memory_allocated(0), current_alloc[0])
        self.assertTrue(
            all(
                torch.xpu.memory_allocated(idx) == current_alloc[idx]
                for idx in range(1, device_count)
            )
        )
        del a

    def test_memory_stats(self):
        gc.collect()
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.reset_accumulated_memory_stats()
        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        self.assertEqual(prev_allocated, prev_max_allocated)
        self.assertEqual(prev_reserved, prev_max_reserved)
        # Activate 1kB memory
        prev_active_current = torch.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        tmp = torch.randn(256, device="xpu")
        # Detect if the current active memory is 1kB
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(torch.accelerator.memory_stats()["active_bytes.all.freed"], 0)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.freed"], 1024
        )
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(torch.accelerator.max_memory_reserved(), prev_max_reserved)

    @unittest.skipIf(
        int(torch.version.xpu) < 20250000,
        "Test requires SYCL compiler version 2025.0.0 or newer.",
    )
    def test_mem_get_info(self):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        before_free_bytes, before_total_bytes = torch.xpu.mem_get_info()
        # increasing to 1MB to force acquiring a new block.
        torch.randn(1024 * 256, device="xpu")
        torch.xpu.synchronize()
        after_free_bytes, after_total_bytes = torch.xpu.mem_get_info()

        self.assertGreaterEqual(before_free_bytes, after_free_bytes)
        self.assertEqual(before_total_bytes, after_total_bytes)

    def test_get_arch_list(self):
        arch_list = torch.xpu.get_arch_list()
        if not arch_list:
            return
        flags = torch.xpu.get_gencode_flags()
        for arch in arch_list:
            self.assertTrue(arch in flags)

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_can_device_access_peer(self):
        device_count = torch.xpu.device_count()
        for device in range(device_count):
            for peer in range(device_count):
                self.assertEqual(
                    torch.xpu.can_device_access_peer(device, peer),
                    torch.xpu.can_device_access_peer(peer, device),
                )

    @serialTest()
    def test_memory_snapshot(self):
        def _test_memory_stats_generator(N=35):
            m0 = torch.xpu.memory_allocated()

            def alloc(*size):
                return torch.empty(*size, device="xpu")

            yield

            tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
            m1 = torch.xpu.memory_allocated()
            yield

            tensors2 = []

            for i in range(1, int(N / 2) + 1):
                # small ones
                tensors2.append(alloc(i, i * 4))
                yield

            for i in range(5, int(N / 2) + 5):
                # large ones
                tensors2.append(alloc(i, i * 7, i * 9, i * 11))
                yield

            tensors2.append(alloc(0, 0, 0))
            yield

            permute = []
            for i in torch.randperm(len(tensors2)):
                permute.append(tensors2[i])
                yield

            del tensors2
            yield
            tensors2 = permute
            yield
            del permute
            yield

            for i in range(int(N / 2)):
                del tensors2[i]
                yield

            for i in range(2, int(2 * N / 3) + 2):
                tensors2.append(alloc(i, i * 3, i * 8))
                yield

            del tensors2
            self.assertEqual(torch.xpu.memory_allocated(), m1)
            yield True

            del tensors1
            self.assertEqual(torch.xpu.memory_allocated(), m0)

        def _check_memory_stat_consistency():
            snapshot = torch.xpu.memory_snapshot()

            expected_each_device = collections.defaultdict(
                lambda: collections.defaultdict(int)
            )

            for segment in snapshot:
                expandable = segment["is_expandable"]
                expected = expected_each_device[segment["device"]]
                pool_str = segment["segment_type"] + "_pool"

                if not expandable:
                    expected["segment.all.current"] += 1
                    expected[f"segment.{pool_str}.current"] += 1

                expected["allocated_bytes.all.current"] += segment["allocated_size"]
                expected[f"allocated_bytes.{pool_str}.current"] += segment[
                    "allocated_size"
                ]

                expected["reserved_bytes.all.current"] += segment["total_size"]
                expected[f"reserved_bytes.{pool_str}.current"] += segment["total_size"]

                expected["active_bytes.all.current"] += segment["active_size"]
                expected[f"active_bytes.{pool_str}.current"] += segment["active_size"]

                expected["requested_bytes.all.current"] += segment["requested_size"]
                expected[f"requested_bytes.{pool_str}.current"] += segment[
                    "requested_size"
                ]

                sum_requested = 0
                is_split = len(segment["blocks"]) > 1
                for block in segment["blocks"]:
                    if block["state"] == "active_allocated":
                        expected["allocation.all.current"] += 1
                        expected[f"allocation.{pool_str}.current"] += 1

                    if block["state"].startswith("active_"):
                        sum_requested += block["requested_size"]
                        expected["active.all.current"] += 1
                        expected[f"active.{pool_str}.current"] += 1

                    if block["state"] == "inactive" and is_split and not expandable:
                        expected["inactive_split.all.current"] += 1
                        expected[f"inactive_split.{pool_str}.current"] += 1
                        expected["inactive_split_bytes.all.current"] += block["size"]
                        expected[f"inactive_split_bytes.{pool_str}.current"] += block[
                            "size"
                        ]

                self.assertEqual(sum_requested, segment["requested_size"])

            for device, expected in expected_each_device.items():
                stats = torch.xpu.memory_stats(device)
                for k, v in expected.items():
                    self.assertEqual(v, stats[k])

        gc.collect()
        torch.xpu.empty_cache()
        for _ in _test_memory_stats_generator():
            _check_memory_stat_consistency()

    @unittest.skipUnless(IS_X86 and IS_LINUX, "x86 linux only cpp unwinding")
    def test_direct_traceback(self):
        from torch._C._profiler import gather_traceback, symbolize_tracebacks

        c = gather_traceback(True, True, True)
        (r,) = symbolize_tracebacks([c])
        r = str(r)
        self.assertTrue("test_xpu.py" in r)
        self.assertTrue("unwind" in r)

    def test_memory_snapshot_with_python(self):
        try:
            gc.collect()
            torch.xpu.memory.empty_cache()
            torch.xpu.memory._record_memory_history("state", stacks="python")
            # Make x the second block in a segment
            torch.rand(2 * 311, 411, device="xpu")
            unused = torch.rand(310, 410, device="xpu")
            x = torch.rand(311, 411, device="xpu")

            # Allocate many 512B tensors to fill a segment and test history merging.
            tensors = [torch.rand(128, device="xpu") for _ in range(1000)]
            while tensors:
                del tensors[random.randint(0, len(tensors) - 1)]

            torch.rand(128 * 5, device="xpu")

            ss = torch._C._xpu_memorySnapshot(None)
            found_it = False
            for seg in ss["segments"]:
                self.assertTrue("frames" in seg)
                for b in seg["blocks"]:
                    # Look for x by size
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("test_xpu" in b["frames"][0]["filename"])
                        found_it = True
                        self.assertEqual(x.untyped_storage().data_ptr(), b["address"])
            self.assertTrue(found_it)

            del unused
            del x
            gc.collect()
            torch.xpu.empty_cache()
            ss = torch._C._xpu_memorySnapshot(None)
            self.assertTrue(
                ss["device_traces"][0][-1]["action"]
                in ("segment_free", "segment_unmap")
            )

        finally:
            torch.xpu.memory._record_memory_history(None)

    @unittest.skipUnless(IS_X86 and IS_LINUX, "x86 linux only cpp unwinding")
    def test_memory_snapshot_with_cpp(self):
        try:
            gc.collect()
            torch.xpu.memory.empty_cache()
            torch.xpu.memory._record_memory_history("state", stacks="all")
            _ = torch.rand(311, 411, device="xpu")

            ss = torch.xpu.memory.memory_snapshot()
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("::rand" in str(b["frames"]))
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.xpu.memory._record_memory_history(None)

    @unittest.skipUnless(IS_X86 and IS_LINUX, "x86 linux only cpp unwinding")
    def test_memory_plots_free_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                gc.collect()
                torch.xpu.memory.empty_cache()
                torch.xpu.memory._record_memory_history(context=context)
                x = None

                def thealloc():
                    nonlocal x
                    x = torch.rand(3, 4, device="xpu")

                def thefree():
                    nonlocal x
                    del x

                thealloc()
                thefree()
                ss = json.dumps(torch.xpu.memory._snapshot())
                self.assertEqual(("thefree" in ss), (context == "all"))
                self.assertEqual(("thealloc" in ss), (context != "state"))
            finally:
                torch.xpu.memory._record_memory_history(None)

    def test_memory_snapshot_script(self):
        try:
            gc.collect()
            torch.xpu.memory.empty_cache()
            torch.xpu.memory._record_memory_history("state", stacks="python")

            @torch.jit.script
            def foo():
                return torch.rand(311, 411, device="xpu")

            _ = foo()

            ss = torch.xpu.memory.memory_snapshot()
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertEqual(b["frames"][0]["name"], "foo")
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.xpu.memory._record_memory_history(None)

    def collect_frames(
        self, augmented_snapshot, collect_device_traces=True, collect_segments=True
    ):
        """Collects all frames that has node metadata from a memory snapshot."""
        # Collect all frames with FX metadata
        fx_frames = []

        # Check device traces for FX debug fields
        if collect_device_traces:
            for trace_list in augmented_snapshot.get("device_traces", []):
                for trace_entry in trace_list:
                    if not isinstance(trace_entry, dict):
                        continue
                    for frame in trace_entry.get("frames", []):
                        if not isinstance(frame, dict):
                            continue
                        if "fx_node_op" in frame or "fx_node_name" in frame:
                            fx_frames.append(frame)

        # Check segments/blocks for FX debug fields
        if collect_segments:
            for segment in augmented_snapshot.get("segments", []):
                for block in segment.get("blocks", []):
                    for frame in block.get("frames", []):
                        if not isinstance(frame, dict):
                            continue
                        if "fx_node_op" in frame or "fx_node_name" in frame:
                            fx_frames.append(frame)
        return fx_frames

    @torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
    def test_fx_memory_profiler_augmentation(self):
        """Test that memory snapshots are augmented with FX debug information."""

        class MLPModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(5)
                self.net1 = torch.nn.Linear(10, 16, bias=True, device="xpu")
                self.relu = torch.nn.ReLU()
                self.net2 = torch.nn.Linear(16, 10, bias=True, device="xpu")

            def forward(self, x):
                a = self.net1(x)
                b = self.relu(a)
                c = self.net2(b)
                return c

        class MLPModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(5)
                self.net1 = torch.nn.Linear(10, 16, bias=True, device="xpu")
                self.relu = torch.nn.ReLU()
                self.net2 = torch.nn.Linear(16, 10, bias=True, device="xpu")

            def forward(self, x):
                d = self.net1(x)
                e = self.relu(d)
                f = self.net2(e)
                return f

        if self.expandable_segments:
            self.skipTest(
                "Requires driver update to fix oneDNN primitive operations when using expandable segments."
            )
        mod = MLPModule()
        gc.collect()
        torch.xpu.memory.empty_cache()
        torch.xpu.memory._record_memory_history()
        compiled = torch.compile(mod, backend="aot_eager", fullgraph=True)
        _ = compiled(torch.randn(10, 10, device="xpu"))
        augmented_snapshot = torch.xpu.memory._snapshot(augment_with_fx_traces=True)
        torch.xpu.memory._record_memory_history(enabled=None, clear_history=True)
        gc.collect()
        torch.xpu.empty_cache()

        fx_frames = self.collect_frames(augmented_snapshot)
        self.assertGreater(len(fx_frames), 2)

        for frame in fx_frames:
            # Every FX frame should have both node_op and node_name
            self.assertIn("fx_node_op", frame)
            self.assertIn("fx_node_name", frame)
            self.assertIn("fx_node_target", frame)
            self.assertIn("fx_original_trace", frame)

            self.assertIn(frame["fx_node_name"], ["addmm", "relu", "addmm_1"])
            fx_node_name = frame["fx_node_name"]
            if fx_node_name == "addmm":
                self.assertIn("a = self.net1(x)", frame["fx_original_trace"])
            elif fx_node_name == "addmm_1":
                self.assertIn("c = self.net2(b)", frame["fx_original_trace"])
            elif fx_node_name == "relu":
                self.assertIn("b = self.relu(a)", frame["fx_original_trace"])

        # Test that when we have two graphs with the same src_code, they're not hashed
        # to the same metadata
        mod = MLPModule2()
        torch.xpu.memory._record_memory_history()
        compiled = torch.compile(mod, backend="aot_eager", fullgraph=True)
        _ = compiled(torch.randn(10, 10, device="xpu"))
        augmented_snapshot = torch.xpu.memory._snapshot(augment_with_fx_traces=True)
        torch.xpu.memory._record_memory_history(enabled=None, clear_history=True)

        # avoid collecting segments from previous run for unit test purpose
        fx_frames = self.collect_frames(augmented_snapshot, collect_segments=False)
        self.assertGreater(len(fx_frames), 0)

        for frame in fx_frames:
            # Every FX frame should have both node_op and node_name
            self.assertIn("fx_node_op", frame)
            self.assertIn("fx_node_name", frame)
            self.assertIn("fx_node_target", frame)
            self.assertIn("fx_original_trace", frame)

            self.assertIn(frame["fx_node_name"], ["addmm", "relu", "addmm_1"])
            fx_node_name = frame["fx_node_name"]
            if fx_node_name == "addmm":
                self.assertIn("d = self.net1(x)", frame["fx_original_trace"])
            elif fx_node_name == "addmm_1":
                self.assertIn("f = self.net2(e)", frame["fx_original_trace"])
            elif fx_node_name == "relu":
                self.assertIn("e = self.relu(d)", frame["fx_original_trace"])

    def get_dummy_allocator(self, check_vars):
        dummy_allocator_source_vars = """
        #include <torch/extension.h>
        #include <c10/xpu/XPUFunctions.h>

        extern "C" {
          C10_EXPORT int called_dummy_alloc = 0;
          C10_EXPORT int called_dummy_free = 0;

          C10_EXPORT void* dummy_alloc(size_t size, int device, sycl::queue* queue) {
            called_dummy_alloc = 123;
            auto& sycl_device = c10::xpu::get_raw_device(device);
            auto& sycl_context = c10::xpu::get_device_context();
            void* ptr = sycl::malloc_shared(size, sycl_device, sycl_context);
            return ptr;
          }

          C10_EXPORT void dummy_free(void* ptr, size_t size, int device, sycl::queue* queue) {
            called_dummy_free = 321;
            sycl::free(ptr, c10::xpu::get_device_context());
          }
        }
        """
        dummy_allocator_source_no_vars = """
        #include <torch/extension.h>
        #include <c10/xpu/XPUFunctions.h>

        extern "C" {
          C10_EXPORT void* dummy_alloc(size_t size, int device, sycl::queue* queue) {
            auto& sycl_device = c10::xpu::get_raw_device(device);
            auto& sycl_context = c10::xpu::get_device_context();
            void* ptr = sycl::malloc_shared(size, sycl_device, sycl_context);
            return ptr;
          }

          C10_EXPORT void dummy_free(void* ptr, size_t size, int device, sycl::queue* queue) {
            sycl::free(ptr, c10::xpu::get_device_context());
          }
        }
        """

        from torch.utils.cpp_extension import load_inline

        dummy_allocator_libname = "dummy_allocator"
        dummy_allocator = load_inline(
            name=dummy_allocator_libname,
            cpp_sources=dummy_allocator_source_vars
            if check_vars
            else dummy_allocator_source_no_vars,
            is_python_module=False,
            keep_intermediates=False,
            verbose=True,
            with_sycl=True,
        )
        allocator = torch.xpu.memory.XPUPluggableAllocator(
            dummy_allocator,
            "dummy_alloc",
            "dummy_free",
        )
        return allocator, dummy_allocator

    def test_xpu_pluggable_allocator(self):
        torch.xpu.init()
        allocator, dummy_allocator = self.get_dummy_allocator(True)
        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")
        self.assertEqual(called_dummy_alloc.value, 0)
        self.assertEqual(called_dummy_free.value, 0)

        with self.assertRaises(RuntimeError):
            torch.xpu.memory.change_current_allocator(allocator)

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        test_script = """\
import ctypes
import torch
from torch.utils.cpp_extension import load_inline

dummy_allocator_source_vars = \"\"\"\
#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>

extern "C" {
  C10_EXPORT int called_dummy_alloc = 0;
  C10_EXPORT int called_dummy_free = 0;

  C10_EXPORT void* dummy_alloc(size_t size, int device, sycl::queue* queue) {
    called_dummy_alloc = 123;
    auto& sycl_device = c10::xpu::get_raw_device(device);
    auto& sycl_context = c10::xpu::get_device_context();
    void* ptr = sycl::malloc_shared(size, sycl_device, sycl_context);
    return ptr;
  }

  C10_EXPORT void dummy_free(void* ptr, size_t size, int device, sycl::queue* queue) {
    called_dummy_free = 321;
    sycl::free(ptr, c10::xpu::get_device_context());
  }
}
\"\"\"

if __name__ == "__main__":
    dummy_allocator = load_inline(
        name='dummy_allocator',
        cpp_sources=dummy_allocator_source_vars,
        is_python_module=False,
        keep_intermediates=False,
        verbose=True,
        with_sycl=True,
    )

    allocator = torch.xpu.memory.XPUPluggableAllocator(
        dummy_allocator,
        "dummy_alloc",
        "dummy_free",
    )
    torch.xpu.memory.change_current_allocator(allocator)
    tensor = torch.randn(100, device='xpu')
    del tensor
    allocator_lib = ctypes.CDLL(dummy_allocator)
    called_dummy_alloc = ctypes.c_int.in_dll(allocator_lib, "called_dummy_alloc")
    called_dummy_free = ctypes.c_int.in_dll(allocator_lib, "called_dummy_free")
    print(called_dummy_alloc.value, called_dummy_free.value)
"""
        rc = check_output(test_script).splitlines()[-1]
        called_dummy_alloc_value, called_dummy_free_value = rc.split()
        self.assertEqual(called_dummy_alloc_value, "123")
        self.assertEqual(called_dummy_free_value, "321")

    def test_torch_version_xpu(self):
        self.assertEqual(len(torch.version.xpu), 8)
        compiler_version = int(torch.version.xpu)
        self.assertGreater(compiler_version, 20230000)
        if IS_LINUX:
            library = find_library_location("libtorch_xpu.so")
            cmd = f"ldd {library} | grep libsycl"
            results = subprocess.check_output(cmd, shell=True).strip().split(b"\n")
            # There should be only one libsycl.so
            self.assertEqual(len(results), 1)
            for result in results:
                self.assertTrue(b"libsycl.so" in result)

    def test_dlpack_conversion(self):
        if self.expandable_segments:
            self.skipTest("Skipping DLPack test for expandable segments allocator.")
        x = make_tensor((5,), dtype=torch.float32, device="xpu")
        if IS_WINDOWS and int(torch.version.xpu) < 20250000:
            with self.assertRaisesRegex(
                NotImplementedError,
                "Default context is not supported on XPU by default on Windows for SYCL compiler versions earlier than 2025.0.0.",
            ):
                torch.to_dlpack(x)
        else:
            z = torch.from_dlpack(torch.to_dlpack(x))
            z[0] = z[0] + 1.0
            self.assertEqual(z, x)

    def test_graph_is_current_stream_capturing(self):
        self.assertFalse(torch.xpu.is_current_stream_capturing())
        s = torch.xpu.Stream()
        with torch.xpu.stream(s):
            g = torch.xpu.XPUGraph()
            self.assertFalse(torch.xpu.is_current_stream_capturing())
            g.capture_begin()
            self.assertTrue(torch.xpu.is_current_stream_capturing())
            g.capture_end()

    def test_graph_capture_simple(self):
        s = torch.xpu.Stream()

        with torch.xpu.stream(s):
            a = torch.full((1000,), 1, device="xpu")
            g = torch.xpu.XPUGraph()
            torch.xpu.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.xpu.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

    def test_graphsafe_set_get_rng_state(self):
        # Define a function to create generator states, with optional graph registration
        def create_states(generator):
            """Initializes generator states and registers them with a XPU graph if provided."""
            # Ensure the XPU generator is initialized
            torch.rand(1, device="xpu")
            generator.manual_seed(0)

            # Save the current state of the generator
            old_state = generator.graphsafe_get_state()
            # Create and save a cloned state of the generator
            new_state = generator.clone_state()
            # Return the original generator and its two states
            return generator, old_state, new_state

        def register_states_to_graph(generator_state, graph):
            _, old_state, new_state = generator_state
            graph.register_generator_state(old_state)
            graph.register_generator_state(new_state)

        # Define a function to perform specific RNG actions using the generator's states
        def perform_random_generation_steps(generator_state):
            generator, old_state, new_state = generator_state
            random_values = []

            # Generate random numbers with the new generator state
            generator.graphsafe_set_state(new_state)
            random_values.append(torch.rand(5, device="xpu", generator=generator))

            # Generate random numbers twice with the old generator state
            generator.graphsafe_set_state(old_state)
            random_values.extend(
                [torch.rand(5, device="xpu", generator=generator) for _ in range(2)]
            )

            return random_values

        # Define a function to retrieve the final offsets of the original and new generator states
        def get_final_offsets_of_states(generator_state):
            _, old_state, new_state = generator_state
            old_state_offset = old_state.get_offset()
            new_state_offset = new_state.get_offset()
            return old_state_offset, new_state_offset

        # Set up and test a new XPU generator
        generator = torch.Generator(device="xpu")
        generator_state = create_states(generator)

        # Set up and test the default XPU generator with a XPU Graph
        g = torch.xpu.XPUGraph()
        s = torch.xpu.Stream()
        default_generator = torch.xpu.default_generators[0]
        default_generator_state = create_states(default_generator)
        register_states_to_graph(default_generator_state, g)

        # Perform random number generation within a XPU graph
        with torch.xpu.stream(s):
            g.capture_begin()
            graphed_random_values = perform_random_generation_steps(
                default_generator_state
            )
            g.capture_end()

        # Synchronize the streams and replay the graph
        torch.xpu.current_stream().wait_stream(s)
        for _ in range(3):
            random_values = perform_random_generation_steps(generator_state)
            g.replay()
            offset = get_final_offsets_of_states(generator_state)
            graph_offset = get_final_offsets_of_states(default_generator_state)

            # Compare the final offsets of states for both generators to ensure consistency
            self.assertEqual(offset, graph_offset)
            # Compare the states generated outside and inside the graph
            self.assertEqual(random_values, graphed_random_values)

    def test_graph_capture_reset_recapture(self):
        s = torch.xpu.Stream()

        with torch.xpu.stream(s):
            a = torch.full((1000,), 1, device="xpu")
            g = torch.xpu.XPUGraph()
            torch.xpu.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.xpu.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

        g.reset()

        with torch.xpu.stream(s):
            g.capture_begin()
            b.fill_(2.0)
            for _ in range(10):
                b = b + 2
            g.capture_end()
        torch.xpu.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 22000.0)

        g.reset()
        del g

    def test_graph_warn_if_has_zero_nodes(self):
        with warnings.catch_warnings(record=True) as caught:
            g = torch.xpu.XPUGraph()
            s = torch.xpu.Stream()
            with torch.xpu.stream(s):
                g.capture_begin()
                g.capture_end()
        self.assertTrue(any("The XPU Graph is empty" in str(w.message) for w in caught))

    def test_graph_capture_oom(self):
        oom_regex = "out of memory"
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.xpu.graph(torch.xpu.XPUGraph()):
                torch.zeros(2**40, device="xpu")

    def test_repeat_graph_capture_oneDNN_memory(self):
        if self.expandable_segments:
            self.skipTest("oneDNN does not support expandable_segments memory")
        (x, y, z) = 1024, 512, 64
        a = torch.rand((x, y), device="xpu")
        b = torch.rand((y, z), device="xpu")

        # warmup
        torch.mm(a, b)

        free_bytes_before, total_bytes = torch.xpu.mem_get_info()
        used_gb_before = (total_bytes - free_bytes_before) / 1e9

        for _ in range(100):
            torch_graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(torch_graph):
                torch.mm(a, b)
            torch_graph.replay()

        free_bytes_after, _ = torch.xpu.mem_get_info()
        used_gb_after = (total_bytes - free_bytes_after) / 1e9

        self.assertFalse(used_gb_before + 0.1 < used_gb_after)

    def test_graph_rng_functional(self):
        ops_with_kwargs = (
            (torch.nn.functional.dropout, {"p": 0.1}),
            (torch.nn.functional.rrelu, {"training": True}),
        )
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="xpu", dtype=torch.float)

            # Control
            torch.xpu.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(stream):
                torch.xpu.manual_seed(5)

                g = torch.xpu.XPUGraph()
                torch.xpu.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.xpu.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # Do the same operations varying seeds
            seeds = [6, 128, 9999]

            for seed in seeds:
                torch.xpu.manual_seed(seed)
                graph_in.copy_(a)
                for _ in range(3):
                    g.replay()

                # If the random seed was not updated then the graph would
                # generate the same output as in previous check.
                try:
                    self.assertNotEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

                # Now repeat the same operations in non-graphed mode.
                torch.xpu.manual_seed(seed)
                for _ in range(3):
                    eager_out.copy_(a)
                    eager_out = op(eager_out, **kwargs)
                    eager_out = op(eager_out, **kwargs)

                # In the end, graph_out and eager_out must be equal
                # as they went under the same set of operations.
                try:
                    self.assertEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.xpu.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)

    def test_graph_rng_distributions(self):
        size = 10000
        input = torch.rand((size,), device="xpu", dtype=torch.float)
        alloc = torch.empty((size,), device="xpu", dtype=torch.float)

        # Torch ops to test with sample args (tuple) and kwargs (dict)
        torch_with_args = (
            ("bernoulli", (input.clone(),), {}),
            ("normal", (input.clone() + 1, 1.0), {}),
            ("poisson", (input.clone(),), {}),
            ("rand", (size,), {"device": "xpu", "dtype": torch.float}),
            ("randint", (0, 3, (size,)), {"device": "xpu", "dtype": torch.float}),
            ("randn", (size,), {"device": "xpu", "dtype": torch.float}),
        )

        # Tensor methods to test with sample args (tuple)
        tensor_with_args = (
            ("bernoulli_", (input.clone(),)),
            ("cauchy_", ()),
            ("exponential_", ()),
            ("geometric_", (0.3,)),
            ("log_normal_", ()),
            ("normal_", ()),
            ("random_", ()),
            ("uniform_", ()),
        )

        def run(module, op, args, kwargs):
            torch.xpu.manual_seed(5)

            # Each path runs a dummy op to increment the state a bit before creating controls.
            if module == "torch":
                dummy = getattr(torch, op)(*args, **kwargs)
                control1 = getattr(torch, op)(*args, **kwargs)
                control2 = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                control1 = alloc.clone()
                control2 = alloc.clone()
                getattr(dummy, op)(*args)
                getattr(control1, op)(*args)
                getattr(control2, op)(*args)

            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(stream):
                torch.xpu.manual_seed(5)

                g = torch.xpu.XPUGraph()
                torch.xpu.empty_cache()
                if module == "torch":
                    g.capture_begin()
                    t1 = getattr(torch, op)(*args, **kwargs)
                    t2 = getattr(torch, op)(*args, **kwargs)
                    g.capture_end()
                else:
                    t1 = alloc.clone()
                    t2 = alloc.clone()
                    g.capture_begin()
                    getattr(t1, op)(*args)
                    getattr(t2, op)(*args)
                    g.capture_end()
            torch.xpu.current_stream().wait_stream(stream)

            try:
                self.assertNotEqual(control1, t1)
                self.assertNotEqual(control2, t2)
            except Exception as e:
                raise RuntimeError("Failed on " + module + "." + op) from e

            # Set a new seed to check if graph would use it
            for seed in [6, 314, 271]:
                torch.xpu.manual_seed(seed)
                # Runs a dummy op prelude, as for controls, to make sure replay()
                # picks up the dummy op's state increment.
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                    control1 = getattr(torch, op)(*args, **kwargs)
                    control2 = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)
                    getattr(control1, op)(*args)
                    getattr(control2, op)(*args)

                torch.xpu.manual_seed(seed)
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)

                t1.copy_(alloc)
                t2.copy_(alloc)

                # Runs RNG ops that fill t1 and t2.
                g.replay()

                try:
                    self.assertEqual(control1, t1)
                    self.assertEqual(control2, t2)
                except Exception as e:
                    raise RuntimeError("Failed on " + module + "." + op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.xpu.synchronize()

        for op_with_args in torch_with_args:
            run("torch", *op_with_args)

        for meth_with_args in tensor_with_args:
            # Adds an empty dict for kwargs, which none of the Tensor methods use
            run("Tensor", *(meth_with_args + ({},)))

    def test_graph_two_successive(self):
        torch.xpu.empty_cache()

        size = 1000
        kSmallBuffer = 2097152

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.xpu.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.xpu.XPUGraph()
            g1 = torch.xpu.XPUGraph()

            a = torch.ones((size,), device="xpu")

            s.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(s):
                g0_args = (
                    (torch.xpu.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g1.capture_end()
            torch.xpu.current_stream().wait_stream(s)

            # mixes unrelated eager ops with replays
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            if share_mem != "Don't share":
                self.assertEqual(
                    reserved_no_sharing  # noqa: F821
                    - torch.xpu.memory_stats()["reserved_bytes.all.current"],
                    kSmallBuffer,
                )
            else:
                reserved_no_sharing = torch.xpu.memory_stats()[
                    "reserved_bytes.all.current"
                ]

            del a, b, c, g0, g1
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    def test_graph_three_successive(self):
        torch.xpu.empty_cache()

        size = 1000

        s = torch.xpu.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            a = torch.ones((size,), device="xpu")

            g0 = torch.xpu.XPUGraph()
            g1 = torch.xpu.XPUGraph()
            g2 = torch.xpu.XPUGraph()

            s.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(s):
                g0_args = (
                    (torch.xpu.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                c = b + 1
                d = b + 2
                g0.capture_end()

                args = (g0.pool(),) if share_mem == "via pool()" else g0_args

                g1.capture_begin(*args)
                e = c + 3
                del c
                g1.capture_end()

                g2.capture_begin(*args)
                f = d + 4
                g2.capture_end()
            torch.xpu.current_stream().wait_stream(s)

            # Tests that replaying in capture order is valid
            g0.replay()
            g1.replay()
            g2.replay()

            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            # Tests that replaying as g0, g2, g1 is only valid if they don't share a pool
            g0.replay()
            g2.replay()
            g1.replay()

            expect_corruption = share_mem != "Don't share"
            self.assertEqual(
                e.sum().item(), size * (7 + 3) if expect_corruption else size * 5
            )
            self.assertEqual(f.sum().item(), size * 7)

            del a, b, d, e, f, g0, g1, g2
            # Tensors used across streams (a, e, f) were held until just now, so no need to call record_stream on them.
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        kSmallSize = 1048576
        kSmallBuffer = 2097152
        kLargeBuffer = 20971520
        kMinLargeAlloc = 10485760
        kRoundLarge = 2097152

        elem = 4

        cases = (
            (512 // elem, 1, kSmallBuffer, kSmallBuffer, "small_pool"),
            (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "small_pool"),
            ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
            (
                (kMinLargeAlloc - 512) // elem,
                2,
                2 * kLargeBuffer,
                kLargeBuffer,
                "large_pool",
            ),
            (
                (kMinLargeAlloc + 512) // elem,
                3,
                3
                * (
                    kRoundLarge
                    * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)
                ),
                kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                "large_pool",
            ),
        )

        stats_to_check = ("segment.", "reserved_bytes.", "active.", "active_bytes.")

        gc.collect()
        torch.xpu.empty_cache()

        s = torch.xpu.Stream()

        for (
            numel,
            delta_xpuMallocs,
            delta_xpuMalloc_bytes,
            delta_xpuMalloc_bytes_post_del_g,
            pool_string,
        ) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 3  # one from "b" plus a sneaky two from XPUGraph's one-element rng seed and offset holders
                delta_active_bytes = (
                    numel * elem + 1024
                )  # + 1024 for XPUGraph's rng seed and offset holders each
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.xpu.XPUGraph()
            s.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(s):
                a = torch.ones((numel,), device="xpu")

                precapture_stats = torch.xpu.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.xpu.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.xpu.memory_stats()

            expecteds = (
                delta_xpuMallocs,
                delta_xpuMalloc_bytes,
                delta_active_blocks,
                delta_active_bytes,
            )
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]

                    if self.expandable_segments and "segment" in stat:
                        expected = 0
                    if (
                        self.expandable_segments
                        and "reserved" in stat
                        and (numel == cases[3][0] or numel == cases[4][0])
                    ):
                        expected = 2 * kLargeBuffer

                    self.assertEqual(
                        current,
                        expected,
                        "Pre to post capture delta of "
                        + stat
                        + f" = {current}, expected = {expected}, numel = {numel}",
                    )

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.xpu.empty_cache()

            del g
            gc.collect()
            torch.xpu.empty_cache()
            postdel_stats = torch.xpu.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_xpuMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]

                if self.expandable_segments and "segment" in stat:
                    expected = 0
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[3][0]
                ):
                    expected = 2 * kLargeBuffer
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[4][0]
                ):
                    expected = kLargeBuffer

                self.assertEqual(
                    current,
                    expected,
                    "Pre capture to post graph delete delta of "
                    + stat
                    + f" = {current}, expected = {expected}, numel = {numel}",
                )

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    @serialTest()
    def test_graph_checkpoint_preserve_rng_state(self):
        torch.xpu.manual_seed(42)

        def fn(x):
            return x * torch.sigmoid(torch.randn(1, device="xpu"))

        fn(torch.ones(1, device="xpu"))

        torch.xpu.manual_seed(42)
        eager_in = torch.ones(1, device="xpu", requires_grad=True)
        eager_out = torch.utils.checkpoint.checkpoint(
            fn, eager_in, use_reentrant=False, preserve_rng_state=True
        )
        (eager_in_grad,) = torch.autograd.grad(eager_out, eager_in)

        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            graph_in = torch.ones(1, device="xpu", requires_grad=True)
            graph_out = torch.utils.checkpoint.checkpoint(
                fn, graph_in, use_reentrant=False, preserve_rng_state=True
            )
            (graph_in_grad,) = torch.autograd.grad(graph_out, graph_in)

        torch.xpu.manual_seed(42)
        g.replay()

        self.assertEqual(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)

    @serialTest()
    def test_graph_manual_seed_mismatch_raises(self):
        torch.xpu.manual_seed(0)
        g = torch.xpu.XPUGraph()
        with self.assertRaisesRegex(
            RuntimeError,
            "XPUGeneratorImpl::set_current_seed can be called during stream capture only if new seed is the same as the original seed.",  # noqa: B950
        ):
            with torch.xpu.graph(g):
                torch.xpu.manual_seed(1)

    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[unittest.expectedFailure]),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        if self.expandable_segments:
            self.skipTest("oneDNN does not support expandable_segments memory")
        torch.manual_seed(5)
        torch.xpu.manual_seed(5)

        N, D_in, H, D_out = 640, 4096, 2048, 1024

        class MLP1(torch.nn.Module):
            def __init__(self, D_in: int, H: int, D_out: int):
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).xpu()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).xpu()

            def forward(self, input_dict: dict):
                x = input_dict["x"]
                return self.net_2(self.net_1(x))

        class MLP2(torch.nn.Module):
            def __init__(self, D_in: int, H: int, D_out: int):
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).xpu()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).xpu()

            def forward(self, x):
                return self.net_2(self.net_1(x))

        class ParameterlessModule(torch.nn.Module):
            def forward(self, x):
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = MLP1(D_in, H, H).xpu()
            model_section2 = MLP2(H, H, D_out).xpu()
            model_section3 = ParameterlessModule().xpu()
            models.append(
                torch.nn.Sequential(model_section1, model_section2, model_section3)
            )

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        opt_graphed = torch.optim.SGD(model_graphed.parameters(), lr=0.1)
        opt_control = torch.optim.SGD(model_control.parameters(), lr=0.1)

        x = torch.randn(N, D_in, device="xpu")
        h = torch.randn(N, H, device="xpu", requires_grad=True)
        h2 = torch.randn(N, D_out, device="xpu", requires_grad=True)
        unused_input = torch.randn(N, H, device="xpu", requires_grad=True)
        y_pred = torch.randn(N, D_out, device="xpu", requires_grad=True)
        y = torch.randn(N, D_out, device="xpu")

        loss_fn_control = torch.nn.functional.mse_loss
        relu_control = torch.nn.functional.relu

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="xpu", enabled=with_amp, cache_enabled=cache_enabled
        ):
            (
                model_graphed[0],
                model_graphed[1],
                model_graphed[2],
                relu_graphed,
                loss_fn_graphed,
            ) = torch.xpu.make_graphed_callables(
                (
                    model_graphed[0],
                    model_graphed[1],
                    model_graphed[2],
                    relu_control,
                    loss_fn_control,
                ),
                (
                    ({"x": x, "unused_input": unused_input},),
                    (h,),
                    (h2,),
                    (y_pred,),
                    (y_pred, y),
                ),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m, opt, relu, loss_fn in zip(
            (model_graphed, model_control),
            (opt_graphed, opt_control),
            (relu_graphed, relu_control),
            (loss_fn_graphed, loss_fn_control),
        ):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.xpu.manual_seed(5)
            for data, target in zip(real_inputs, real_targets):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="xpu", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    y_pred = m({"x": data, "unused_input": unused_input})["output"]
                    y_pred = relu(y_pred)
                    loss = loss_fn(y_pred, target)
                    loss.backward()
                opt.step()

        for p, pc in zip(model_graphed.parameters(), model_control.parameters()):
            self.assertEqual(p, pc)

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((False, False, True)),
            subtest((True, False, True)),
            subtest((True, True, True)),
            subtest((False, False, False)),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables_parameterless_nograd_module(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        torch.manual_seed(5)
        torch.xpu.manual_seed(5)

        N, D_in, H, _ = 640, 4096, 2048, 1024

        class ParameterlessModule(torch.nn.Module):
            def forward(self, input_dict: dict):
                x = input_dict["x"]
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = ParameterlessModule().xpu()
            models.append(torch.nn.Sequential(model_section1))

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        x = torch.randn(N, D_in, device="xpu", requires_grad=False)
        unused_input = torch.randn(N, H, device="xpu", requires_grad=False)
        y = torch.randn(N, D_in, device="xpu")

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="xpu", enabled=with_amp, cache_enabled=cache_enabled
        ):
            model_graphed[0] = torch.xpu.make_graphed_callables(
                model_graphed[0],
                ({"x": x, "unused_input": unused_input},),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x, requires_grad=True) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m in (model_graphed, model_control):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.xpu.manual_seed(5)
            for data, _ in zip(real_inputs, real_targets):
                with torch.amp.autocast(
                    device_type="xpu", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    m({"x": data, "unused_input": unused_input})["output"]

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    def test_graph_make_graphed_callables_same_pool(self):
        if self.expandable_segments:
            self.skipTest("oneDNN does not support expandable_segments memory")
        torch.manual_seed(5)
        torch.xpu.manual_seed(5)
        models = []
        num_models = 3
        for _ in range(num_models):
            models.append(
                torch.nn.Sequential(
                    torch.nn.Linear(32, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                ).xpu()
            )
        # we will reuse the same pool for all graph captures
        mempool = torch.xpu.graph_pool_handle()
        graphed_models = []
        for model in models:
            x = torch.randn([64, 32], device="xpu")
            graphed_model = deepcopy(model)
            graphed_model = torch.xpu.make_graphed_callables(
                graphed_model, (x,), pool=mempool
            )
            graphed_models.append(graphed_model)

        for model, graphed_model in zip(models, graphed_models):
            x = torch.randn([64, 32], device="xpu")
            y = model(x)
            yg = graphed_model(x)
            l = y.norm()
            lg = yg.norm()
            l.backward()
            lg.backward()

            self.assertEqual(y, yg)
            self.assertEqual(l, lg)
            for p, pg in zip(model.parameters(), graphed_model.parameters()):
                self.assertEqual(p, pg)
                self.assertEqual(p.grad, pg.grad)
                self.assertNotEqual(p.data_ptr(), pg.data_ptr())
                self.assertNotEqual(p.grad.data_ptr(), pg.grad.data_ptr())

    def test_graph_optims_with_explicitly_capturable_param_groups(self):
        n_warmup, n_replay = 3, 2
        for optimizer, second_param_group_capturable in product(
            (
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.ASGD,
                torch.optim.Adamax,
                torch.optim.NAdam,
                torch.optim.RAdam,
                torch.optim.Adadelta,
                torch.optim.RMSprop,
                torch.optim.Rprop,
            ),
            (True, False),
        ):
            ref_p1, param1 = (
                torch.nn.Parameter(torch.ones(1, device="xpu")) for _ in range(2)
            )
            ref_p2, param2 = (
                torch.nn.Parameter(torch.ones(1, device="xpu")) for _ in range(2)
            )
            grads1, grads2 = (
                [torch.randn_like(param1) for _ in range(n_warmup + n_replay)]
                for _ in range(2)
            )
            ref_grads1, ref_grads2 = (
                [t.clone() for t in tensors] for tensors in (grads1, grads2)
            )
            params = [
                {"params": [param1], "capturable": True},
                {"params": [param2], "capturable": second_param_group_capturable},
            ]
            opt = optimizer(params)
            opt_ = optimizer(
                [
                    {"params": [ref_p1], "capturable": False},
                    {"params": [ref_p2], "capturable": False},
                ]
            )

            for i in range(n_warmup + n_replay):
                ref_p1.grad = ref_grads1[i]
                ref_p2.grad = ref_grads2[i]
                opt_.step()

            for i in range(n_warmup):
                param1.grad = grads1[i]
                param2.grad = grads2[i]
                opt.step()

            g = torch.xpu.XPUGraph()
            if not second_param_group_capturable:
                with self.assertRaisesRegex(RuntimeError, "Attempting XPU graph"):
                    with torch.xpu.graph(g):
                        opt.step()
            else:
                with torch.xpu.graph(g):
                    opt.step()

                for i in range(n_replay):
                    param1.grad.copy_(grads1[n_warmup + i])
                    param2.grad.copy_(grads2[n_warmup + i])
                    g.replay()
                self.assertEqual(ref_p1, param1)
                self.assertEqual(ref_p2, param2)

    def test_xpu_graph_error_options(self):
        def fn():
            x = torch.zeros([2000], device="xpu")
            y = x + x + x
            return y

        mem = None

        def raw_malloc():
            global mem
            mem = None
            stream = torch.xpu.Stream()
            try:
                with torch.xpu.stream(stream):
                    mem = torch.xpu.caching_allocator_alloc(1024)
            except BaseException:  # noqa: B036
                if mem is None:
                    return
            try:
                torch.xpu.caching_allocator_delete(mem)
                mem = None
                return None
            except BaseException:  # noqa: B036
                pass

        def throws_on_xpu_event():
            graph = torch.xpu.XPUGraph()
            torch.xpu.synchronize()
            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(stream):
                fn()
            stream.synchronize()
            torch.xpu.current_stream().wait_stream(stream)
            torch.xpu.synchronize()
            try:
                with torch.xpu.graph(graph, stream=stream):
                    out = fn()
                    thread = threading.Thread(target=raw_malloc)
                    thread.start()
                    thread.join()
            except Exception:
                if mem is not None:
                    torch.xpu.caching_allocator_delete(mem)
                return True

            return False

        self.assertFalse(throws_on_xpu_event())

    def test_xpu_graph_raw_graph_keep_graph_false(self):
        graph = torch.xpu.XPUGraph(keep_graph=False)
        x = torch.zeros([2000], device="xpu")
        y = torch.ones([2000], device="xpu")
        with torch.xpu.graph(graph):
            z = x + y

        with self.assertRaisesRegex(
            RuntimeError,
            r"instantiate\(\) is intended to be called by the user only when keep_graph=true",
        ):
            raw_pointer = graph.instantiate()

        with self.assertRaisesRegex(
            RuntimeError,
            r"You cannot access the raw xpuGraph_t instance unless XPUGraph was initialized with keep_graph=true",
        ):
            raw_pointer = graph.raw_xpu_graph()

    def test_xpu_graph_raw_graph_reset_and_recapture(self):
        graph = torch.xpu.XPUGraph(keep_graph=True)
        x = torch.zeros([2000], device="xpu")
        with torch.xpu.graph(graph):
            x += 1.0

        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 1.0))
        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 2.0))
        graph.replay()
        self.assertTrue(torch.all(x == 3.0))

        graph.reset()

        x = torch.zeros([2000], device="xpu")
        with torch.xpu.graph(graph):
            x += 2.0

        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 2.0))
        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 4.0))
        graph.replay()
        self.assertTrue(torch.all(x == 6.0))


@contextlib.contextmanager
def caching_host_allocator_use_host_register(use_xpu_host_register: bool):
    if use_xpu_host_register:
        torch._C._accelerator_setAllocatorSettings(
            "pinned_use_xpu_host_register:True,pinned_num_register_threads:8"
        )
    try:
        yield
    finally:
        if use_xpu_host_register:
            torch._C._accelerator_setAllocatorSettings(
                "pinned_use_xpu_host_register:False"
            )


@contextlib.contextmanager
def caching_host_allocator_use_background_threads(use_background_threads: bool):
    if use_background_threads:
        torch._C._accelerator_setAllocatorSettings("pinned_use_background_threads:True")
    try:
        yield
    finally:
        if use_background_threads:
            torch._C._accelerator_setAllocatorSettings(
                "pinned_use_background_threads:False"
            )


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
class TestCachingHostAllocatorXpuGraph(TestCase):
    @parametrize("use_xpu_host_register", [True, False])
    def test_pin_memory_no_use(self, use_xpu_host_register):
        # A pinned host memory block cannot be reused if it is not deleted
        with caching_host_allocator_use_host_register(use_xpu_host_register):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                data = torch.empty(8, pin_memory=True)
                data2 = torch.empty(8, pin_memory=True)
            self.assertNotEqual(data.data_ptr(), data2.data_ptr())
            del data2

    @parametrize("use_xpu_host_register", [True, False])
    def test_pin_memory_no_use2(self, use_xpu_host_register):
        # A pinned host memory block can be reused if it is deleted
        # and has never been used by copy_
        with caching_host_allocator_use_host_register(use_xpu_host_register):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                data = torch.randn(8).pin_memory()
                data_ptr = data.data_ptr()
                del data
                data2 = torch.randn(8).pin_memory()
                self.assertEqual(data2.data_ptr(), data_ptr)

    @parametrize("use_xpu_host_register", [True, False])
    def test_pin_memory_use(self, use_xpu_host_register):
        # A pinned host memory block cannot be reused if it has been used by copy_
        with caching_host_allocator_use_host_register(use_xpu_host_register):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                data = torch.randn(8).pin_memory()
                data_gpu = torch.randn(8, device="xpu")
                data_gpu.copy_(data, non_blocking=True)
                old_data_ptr = data.data_ptr()
                del data
                data2 = torch.randn(8).pin_memory()
            self.assertNotEqual(data2.data_ptr(), old_data_ptr)

    @parametrize("use_xpu_host_register", [True, False])
    @parametrize("use_background_threads", [True, False])
    @parametrize(
        "use_memory, delete_memory",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_two_graphs(
        self, use_background_threads, use_xpu_host_register, use_memory, delete_memory
    ):
        with (
            caching_host_allocator_use_background_threads(use_background_threads),
            caching_host_allocator_use_host_register(use_xpu_host_register),
        ):
            shared_pool = torch.xpu.graph_pool_handle()
            graph1 = torch.xpu.XPUGraph()
            graph2 = torch.xpu.XPUGraph()

            with torch.xpu.graph(graph1, pool=shared_pool):
                data = torch.randn(8).pin_memory()
                if use_memory:
                    data_gpu = torch.randn(8, device="xpu")
                    data_gpu.copy_(data, non_blocking=True)

                old_data_ptr = data.data_ptr()
                if delete_memory:
                    del data

            with torch.xpu.graph(graph2, pool=shared_pool):
                data2 = torch.randn(8).pin_memory()
                if use_memory:
                    data_gpu = torch.randn(8, device="xpu")
                    data_gpu.copy_(data2, non_blocking=True)

                new_data_ptr = data2.data_ptr()
                if delete_memory:
                    del data2

            if delete_memory and not use_memory:
                self.assertEqual(new_data_ptr, old_data_ptr)
            else:
                self.assertNotEqual(new_data_ptr, old_data_ptr)


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestXpuOptims(TestCase):
    @optims(
        [optim for optim in optim_db if optim.has_capturable_arg],
        dtypes=[torch.float32],
    )
    def test_graph_optims(self, dtype, optim_info):
        device = "xpu"
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        steps_warmup = 3
        steps_train = 2

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            kwargs["lr"] = 0.1
            if optim_cls in (torch.optim.Adam, torch.optim.AdamW):
                kwargs["betas"] = (0.9, 0.99)

            for actually_do_graphs in (True, False):
                params = [
                    torch.randn((i + 5, i + 5), device=device) for i in range(2)
                ] + [torch.randn((), device=device)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]

                # capturable=False
                kwargs["capturable"] = False

                opt = optim_cls(params_control, **kwargs)
                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads[i][j]
                    opt.step()

                # capturable=True
                kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i][j]
                    opt.step()

                if actually_do_graphs:
                    g = torch.xpu.XPUGraph()
                    with torch.xpu.graph(g):
                        opt.step()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads[i + steps_warmup][j])
                        g.replay()
                    else:
                        for j, p in enumerate(params_graphed):
                            p.grad = grads[i + steps_warmup][j]
                        opt.step()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    @optims(
        [
            optim
            for optim in optim_db
            if "fused" in optim.supported_impls and "xpu" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_scaling_fused_optimizers(self, dtype, optim_info):
        device = "xpu"
        optim_cls = optim_info.optim_cls

        steps_warmup = 3
        steps_train = 2

        optim_inputs = optim_info.optim_inputs_func(device=device)

        for optim_input in optim_inputs:
            kwargs = optim_input.kwargs
            kwargs["fused"] = True

            for actually_do_graphs in (
                (True, False) if optim_info.has_capturable_arg else (True,)
            ):
                params = [torch.randn((i + 5, i + 5), device=device) for i in range(2)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                # `GradScaler` in-place updates gradients thus it's necessary to duplicate gradients.
                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]
                with torch.no_grad():
                    grads_control = [[g.clone() for g in gs] for gs in grads]
                    grads_graphed = [[g.clone() for g in gs] for gs in grads]

                # Gradient Scaler
                scaler_for_control = torch.amp.GradScaler("xpu", init_scale=128.0)
                with torch.no_grad():
                    scaler_for_control._lazy_init_scale_growth_tracker(device)

                scaler_for_graphed = torch.amp.GradScaler("xpu")
                scaler_for_graphed.load_state_dict(scaler_for_control.state_dict())
                with torch.no_grad():
                    scaler_for_graphed._lazy_init_scale_growth_tracker(device)

                # capturable=False
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = False
                opt = optim_cls(params_control, **kwargs)

                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads_control[i][j]
                    scaler_for_control.step(opt)
                    scaler_for_control.update()

                # capturable=True
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads_graphed[i][j]
                    scaler_for_graphed.step(opt)
                    scaler_for_graphed.update()

                if actually_do_graphs:
                    g = torch.xpu.XPUGraph()
                    with torch.xpu.graph(g):
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads_graphed[i + steps_warmup][j])
                        g.replay()
                    else:
                        for j, p in enumerate(params_graphed):
                            p.grad = grads_graphed[i + steps_warmup][j]
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    @parametrize("foreach, fused", [(False, False), (True, False), (False, True)])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_grad_scaling(self, dtype, optim_info, foreach, fused):
        device = "xpu"
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler(device="xpu", init_scale=4.0)
        g = torch.xpu.XPUGraph()
        s = torch.xpu.Stream()

        weight = torch.ones((100,), device="xpu", requires_grad=True)
        opt = optim_info.optim_cls([weight], lr=0.1, foreach=foreach, fused=fused)
        static_input = torch.ones_like(weight)
        static_grad = torch.ones_like(weight)

        # warmup
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
        torch.xpu.current_stream().wait_stream(s)

        opt.zero_grad(set_to_none=True)

        # capture
        with torch.xpu.stream(s):
            g.capture_begin()
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
            g.capture_end()

        input_vals = [5, 20000, 5, 40000]
        expected_scales = [4, 2, 2, 1]
        expected_growth_trackers = [1, 0, 1, 0]
        expected_grad_vals = [5 * 4, float("inf"), 5 * 2, float("inf")]

        for data, scale, growth_tracker, grad_val in zip(
            input_vals, expected_scales, expected_growth_trackers, expected_grad_vals
        ):
            static_input.fill_(data)
            g.replay()
            self.assertEqual(weight.grad, torch.full_like(weight.grad, grad_val))
            scaler.step(opt)
            scaler.update()
            self.assertEqual(scaler._scale, scale)
            self.assertEqual(scaler._growth_tracker, growth_tracker)


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
class TestXpuOps(TestCase):
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

    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
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


instantiate_device_type_tests(TestXpuOps, globals(), only_for="xpu", allow_xpu=True)


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
class TestXpuAutocast(TestAutocast):
    # These operators are not implemented on XPU backend and we can NOT fall back
    # them to CPU. So we have to skip them at this moment.
    # TODO: remove these operators from skip list when they are implemented on XPU backend.
    # lstm_cell: The operator 'aten::_thnn_fused_lstm_cell' is not currently implemented for the XPU device
    skip_list = ["gru_cell", "lstm_cell"]

    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device("xpu"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    def test_autocast_torch_fp16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if op in self.skip_list:
                skip_test = True  # skip unimplemented op
            if len(op_with_args) == 3:
                skip_test = True  # skip cudnn op
            if not skip_test:
                self._run_autocast_outofplace(
                    op, args, torch.float16, device="xpu", amp_dtype=torch.float16
                )

    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if op in self.skip_list:
                skip_test = True  # skip unimplemented op
            if len(op_with_args) == 3:
                skip_test = True  # skip cudnn op
            if not skip_test:
                self._run_autocast_outofplace(op, args, torch.bfloat16, device="xpu")

    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(
                op, args, torch.float32, device="xpu", amp_dtype=torch.float16
            )

    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="xpu",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    def test_autocast_checkpointing(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).xpu()
        input = torch.rand(
            (8, 8), device="xpu", dtype=torch.float16, requires_grad=True
        )
        for reentrant in (True, False):
            with torch.autocast("xpu"):
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            self.assertTrue(output.requires_grad)
            self.assertTrue(output.dtype is torch.float16)
            output.sum().backward()

    def test_xpu_autocast_dtype(self):
        dtype = torch.get_autocast_dtype("xpu")
        self.assertEqual(dtype, torch.float16)
        mat0_fp32 = torch.randn((10, 10), dtype=torch.float32, device="xpu")
        mat1_fp32 = torch.randn((10, 10), dtype=torch.float32, device="xpu")
        with torch.amp.autocast("xpu"):
            result = torch.mm(mat0_fp32, mat1_fp32)
            self.assertEqual(result.dtype, torch.float16)


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
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


class TestXPUAPISanity(TestCase):
    def test_is_bf16_supported(self):
        self.assertEqual(
            torch.xpu.is_bf16_supported(including_emulation=True),
            torch.xpu.is_available(),
        )

    def test_is_tf32_supported(self):
        if not torch.xpu.is_available():
            self.assertFalse(torch.xpu.is_tf32_supported())

    def test_get_arch_list(self):
        if not torch.xpu._is_compiled():
            self.assertEqual(len(torch.xpu.get_arch_list()), 0)

    def test_torch_config_for_xpu(self):
        config = torch.__config__.show()
        value = re.search(r"USE_XPU=([^,]+)", config)
        self.assertIsNotNone(value)
        if torch.xpu._is_compiled():
            self.assertTrue(value.group(1) in ["ON", "1"])
            value = re.search(r"USE_XCCL=([^,]+)", config)
            if torch.distributed.is_xccl_available():
                self.assertTrue(value.group(1) in ["ON", "1"])
            else:
                self.assertTrue(value.group(1) in ["OFF", "0"])
        else:
            self.assertTrue(value.group(1) in ["OFF", "0"])
            self.assertFalse(torch.distributed.is_xccl_available())
            value = re.search(r"USE_XCCL=([^,]+)", config)
            self.assertIsNotNone(value)
            self.assertTrue(value.group(1) in ["OFF", "0"])


@unittest.skipIf(not TEST_XPU, "XPU not available, skipping tests")
class TestMemPool(TestCase):
    def test_mempool_id(self):
        pool1 = torch.xpu.MemPool().id
        pool2 = torch.xpu.MemPool().id

        # first value of id in a user created pool is always zero
        self.assertEqual(pool1[0] == 0, pool2[0] == 0)

        # each call to torch.xpu.MemPool()
        # increments the id
        self.assertTrue(abs(pool2[1] - pool1[1]) > 0)

    def test_mempool_multithread(self):
        pool_ids = []

        def create_mempool_and_make_active():
            pool = torch.xpu.MemPool()
            pool_ids.extend([pool.id])

        num_threads = 4
        threads = [
            threading.Thread(target=create_mempool_and_make_active)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # each thread should create a unique mempool, since
        # mempool id creation is atomic
        self.assertEqual(len(set(pool_ids)), 4)

    def test_mempool_empty_cache(self):
        torch.xpu.empty_cache()
        pool = torch.xpu.MemPool()
        x = torch.empty(1024, 1024, device="xpu")

        with torch.xpu.use_mem_pool(pool):
            y = torch.empty(1024, 1024, device="xpu")

        del x
        del y
        peak_reserved = torch.xpu.memory_reserved()
        del pool
        after_pool_release = torch.xpu.memory_reserved()

        self.assertTrue(after_pool_release < peak_reserved)
        self.assertTrue(after_pool_release > 0)


instantiate_parametrized_tests(TestXpu)
instantiate_parametrized_tests(TestCachingHostAllocatorXpuGraph)
instantiate_device_type_tests(TestXpuOptims, globals())

if __name__ == "__main__":
    run_tests()
