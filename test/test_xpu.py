# Owner(s): ["module: intel"]

import collections
import ctypes
import gc
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import unittest

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
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    run_tests,
    serialTest,
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

    def test_stream_compatibility(self):
        s1 = torch.xpu.Stream()
        s2 = torch.xpu.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s1.stream_id)
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s2.stream_id)
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.current_stream(torch.accelerator.device_count())

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


if __name__ == "__main__":
    run_tests()
