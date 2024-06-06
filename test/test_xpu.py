# Owner(s): ["module: intel"]

import collections
import sys
import tempfile
import unittest

import torch
import torch.xpu._gpu_trace as gpu_trace
from torch.testing._internal.autocast_test_lists import AutocastTestLists
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
        self.assertEqual(
            device_properties.driver_version, device_capability["driver_version"]
        )
        self.assertEqual(device_properties.has_fp16, device_capability["has_fp16"])
        self.assertEqual(device_properties.has_fp64, device_capability["has_fp64"])
        self.assertEqual(
            device_properties.has_atomic64, device_capability["has_atomic64"]
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

    def test_generic_stream_event(self):
        stream = torch.Stream("xpu")
        self.assertEqual(stream.device_index, torch.xpu.current_device())
        xpu_stream = torch.xpu.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertEqual(stream.stream_id, xpu_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.xpu.current_stream().stream_id)

        event1 = torch.Event("xpu")
        event2 = torch.Event("xpu")
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
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_xpu.cpu(), a + b)
        with self.assertRaisesRegex(
            NotImplementedError, "elapsedTime is not supported by XPU backend."
        ):
            event1.elapsed_time(event2)

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


instantiate_device_type_tests(TestXpu, globals(), only_for="xpu")


class TestXpuAutocast(TestCase):
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device("xpu"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    def _run_autocast_outofplace(
        self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None
    ):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}
        fast_dtype = torch.bfloat16 if run_as_type == torch.bfloat16 else torch.float16
        self.assertFalse(torch.is_autocast_enabled())
        with torch.amp.autocast("xpu", dtype=fast_dtype):
            self.assertTrue(torch.is_autocast_enabled())

            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(
                        out_type == output.dtype,
                        f"autocast for torch.{op} produced {output.dtype}, should produce {out_type}",
                    )

            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(
                        out_type == output_method.dtype,
                        f"autocast for torch.{op} produced {output_method.dtype}, should produce torch.{out_type}",
                    )

            self.assertTrue(
                (output is not None) or (output_method is not None),
                f"{op} not found as an attribute on either Tensor or the requested module {module}",
            )

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(
                    comparison, f"torch.{op} result did not match Tensor.{op} result"
                )

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.amp.autocast("xpu", enabled=False):
                self.assertFalse(torch.is_autocast_enabled())

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(
                        *cast(args, run_as_type), **add_kwargs
                    )
                else:
                    control = getattr(args[0].to(run_as_type), op)(
                        *cast(args[1:], run_as_type), **add_kwargs
                    )
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, f"torch.{op} result did not match control")
            self.assertTrue(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled())

    def test_autocast_torch_fp16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if len(op_with_args) == 3:
                skip_test = True  # skip cudnn op
            if not skip_test:
                self._run_autocast_outofplace(op, args, torch.float16)

    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if len(op_with_args) == 3:
                skip_test = True  # skip cudnn op
            if not skip_test:
                self._run_autocast_outofplace(op, args, torch.bfloat16)

    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

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
