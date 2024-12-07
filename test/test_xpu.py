# Owner(s): ["module: intel"]

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
    onlyXPU,
    OpDTypes,
    ops,
    skipXPUIf,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_LINUX,
    IS_WINDOWS,
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint_sequential


if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

 
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
test_multi_process(model, input)
test_multi_process(model, input)
print(torch.xpu.device_count())
"""
        rc = check_output(test_script)
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
            start_event.elapsed_time(end_event)
        else:
            with self.assertRaisesRegex(
                NotImplementedError,
                "elapsed_time of XPUEvent requires PyTorch to be built with SYCL compiler version 2025.0.0 or newer.",
            ):
                start_event.elapsed_time(end_event)

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
            event1.elapsed_time(event2)
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
    @ops(_xpu_computation_ops, dtypes=OpDTypes.any_common_cpu_gpu_one)
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
        tensor = torch.zeros(1024, device="xpu")

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 800000000.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="xpu")

        with self.assertRaisesRegex(RuntimeError, "XPU out of memory."):
            torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="xpu")

    def test_raises_oom(self):
        torch.xpu.memory.empty_cache()
        with self.assertRaises(torch.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="xpu")

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
        x = torch.ones(10, device="xpu:0")
        self.assertGreater(torch.xpu.memory_allocated(0), current_alloc[0])
        self.assertTrue(
            all(
                torch.xpu.memory_allocated(idx) == current_alloc[idx]
                for idx in range(1, device_count)
            )
        )

    @skipXPUIf(
        int(torch.version.xpu) < 20250000,
        "Test requires SYCL compiler version 2025.0.0 or newer.",
    )
    def test_mem_get_info(self):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        before_free_bytes, before_total_bytes = torch.xpu.mem_get_info()
        # increasing to 1MB to force acquiring a new block.
        t = torch.randn(1024 * 256, device="xpu")
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

    def test_torch_version_xpu(self):
        self.assertEqual(len(torch.version.xpu), 8)
        compiler_version = int(torch.version.xpu)
        self.assertGreater(compiler_version, 20230000)
        if IS_LINUX:
            library = find_library_location("libtorch_xpu.so")
            cmd = f"ldd {library} | grep libsycl"
            results = subprocess.check_output(cmd, shell=True).strip().split(b"\n")
            # There should be only one libsycl.so or libsycl-preview.so
            self.assertEqual(len(results), 1)
            for result in results:
                if b"libsycl.so" in result:
                    self.assertGreaterEqual(compiler_version, 20250000)
                elif b"libsycl-preview.so" in result:
                    self.assertLess(compiler_version, 20250000)
                else:
                    self.fail("Unexpected libsycl library")

    def test_dlpack_conversion(self):
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


instantiate_device_type_tests(TestXpu, globals(), only_for="xpu", allow_xpu=True)


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
