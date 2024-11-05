# Owner(s): ["oncall: profiler"]

# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turnning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

import json
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import _dynamo as torchdynamo
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.profiler import (
    ExecutionTraceObserver,
    kineto_available,
    profile,
    record_function,
    supported_activities,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfHpu,
    skipIfTorchDynamo,
    TEST_HPU,
    TEST_XPU,
    TestCase,
)
from torch.utils._triton import has_triton


Json = Dict[str, Any]


class TestExecutionTrace(TestCase):
    def payload(self, device, use_device=False):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with record_function("## TEST 1 ##", "1, 2, 3"):
            inf_val = float("inf")
            neg_inf_val = float("-inf")
            nan_val = float("nan")
            rf_handle = _record_function_with_args_enter(
                "## TEST 2 ##",
                1,
                False,
                2.5,
                [u, u],
                (u, u),
                "hello",
                u,
                inf_val,
                neg_inf_val,
                nan_val,
            )
            x = torch.randn(10, 10, requires_grad=True)
            if use_device:
                x = x.to(device)
            y = torch.randn(10, 10, requires_grad=True)
            if use_device:
                y = y.to(device)
            z = x + y + x * y + x * y
            z.backward(z)
            gelu = nn.GELU()
            m = torch.randn(2)
            _ = gelu(m)
            if use_device:
                z = z.cpu()
            _record_function_with_args_exit(rf_handle)

    def get_execution_trace_root(self, output_file_name) -> Json:
        nodes = []
        with open(output_file_name) as f:
            et_graph = json.load(f)
            assert "nodes" in et_graph
            nodes = et_graph["nodes"]
        return nodes

    def get_execution_trace_rf_ids(self, nodes: List[Json]) -> List[int]:
        """Returns a sorted list of rf_id (record function ids) in execution trace"""

        def get_rf_id(node):
            attrs = node["attrs"]
            for a in attrs:
                if a["name"] == "rf_id":
                    return a["value"]
            return None

        rf_ids_ = (
            get_rf_id(n)
            for n in nodes
            if n["name"] != "[pytorch|profiler|execution_trace|process]"
            and n["name"] != "[pytorch|profiler|execution_trace|thread]"
        )
        return sorted(rf_id for rf_id in rf_ids_ if rf_id is not None)

    def get_kineto_rf_ids(self, events: List[Json]) -> List[int]:
        """Returns a sorted list of Record function IDs for CPU operators and user annotations"""
        ops_and_annotations = (
            e for e in events if e.get("cat", "") in ["cpu_op", "user_annotation"]
        )
        return sorted(
            e.get("args", {}).get("Record function id", -1) for e in ops_and_annotations
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfHpu
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_execution_trace_with_kineto(self, device):
        trace_called_num = 0

        def trace_handler(p):
            nonlocal trace_called_num
            trace_called_num += 1

        use_device = (
            torch.profiler.ProfilerActivity.CUDA
            or torch.profiler.ProfilerActivity.XPU in supported_activities()
            or torch.profiler.ProfilerActivity.HPU in supported_activities()
        )
        # Create a temp file to save execution trace and kineto data.
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        kt = tempfile.NamedTemporaryFile(
            mode="w+t", suffix=".kineto.json", delete=False
        )
        kt.close()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            on_trace_ready=trace_handler,
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback(fp.name)
            ),
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(device, use_device=use_device)
                p.step()
            self.assertEqual(fp.name, p.execution_trace_observer.get_output_file_path())

        # Uncomment for debugging
        # print("Output kineto = ", kt.name)
        # print("Output ET = ", fp.name)

        p.export_chrome_trace(kt.name)
        self.assertEqual(trace_called_num, 1)

        nodes = self.get_execution_trace_root(fp.name)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        self.assertTrue(found_root_node)
        # Since profiler trace is active for 2 iterations
        self.assertEqual(loop_count, 2)

        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func ID (rf_id) and External IDs
        # both of these should match for the same trace window.

        with open(kt.name) as f:
            kineto = json.load(f)
            events = kineto["traceEvents"]

        # Look up rf_ids in both Execution and Kineto trace as two lists.
        rf_ids_et = self.get_execution_trace_rf_ids(nodes)
        rf_ids_kineto = self.get_kineto_rf_ids(events)

        self.assertCountEqual(rf_ids_et, rf_ids_kineto)
        self.assertListEqual(
            rf_ids_et,
            rf_ids_kineto,
            msg=f"ET and kineto rf_id should exactly match\n"
            f"  rf_ids_et = {rf_ids_et}\n"
            f"  rf_ids_kineto = {rf_ids_kineto}\n",
        )

    def test_execution_trace_alone(self, device):
        use_device = (
            torch.profiler.ProfilerActivity.CUDA
            or torch.profiler.ProfilerActivity.HPU in supported_activities()
            or torch.profiler.ProfilerActivity.XPU in supported_activities()
        )
        # Create a temp file to save execution trace data.
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        expected_loop_events = 0

        et = ExecutionTraceObserver().register_callback(fp.name)

        et.start()
        for idx in range(5):
            expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)
        et.stop()

        assert fp.name == et.get_output_file_path()
        et.unregister_callback()
        nodes = self.get_execution_trace_root(fp.name)
        loop_count = 0
        # Expected tensor object tuple size, in th form of:
        # [tensor_id, storage_id, offset, numel, itemsize, device_str]
        tensor_tuple_size = 6
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
            # Check if tensor tuple representation size is correct.
            if n["name"] == "## TEST 2 ##":
                assert len(n["inputs"]["values"][3][0]) == tensor_tuple_size
        assert found_root_node
        assert loop_count == expected_loop_events

    @unittest.skipIf(IS_WINDOWS, "torch.compile does not support WINDOWS")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(
        (not has_triton()) or (not TEST_CUDA and not TEST_XPU),
        "need triton and device(CUDA or XPU) availability to run",
    )
    def test_execution_trace_with_pt2(self, device):
        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        a, b, c = (torch.randn(4, 4, requires_grad=True).to(device) for _ in range(3))

        inputs = [a, b, c]
        with torch._inductor.config.patch(compile_threads=1):
            fn(*inputs)

        # Create a temp file to save execution trace data.
        fp = tempfile.NamedTemporaryFile("w+t", suffix="_et.json", delete=False)
        fp.close()

        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback(fp.name)
            ),
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    fn(*inputs)
                p.step()

        nodes = self.get_execution_trace_root(fp.name)
        found_captured_triton_kernel_node = False
        for n in nodes:
            assert "name" in n
            if "triton_" in n["name"]:
                for attr in n["attrs"]:
                    if attr["name"] == "kernel_file" and attr["value"] != "":
                        found_captured_triton_kernel_node = True
                        assert len(n["inputs"]["values"]) > 0
                        assert len(n["outputs"]["values"]) == 0
        assert found_captured_triton_kernel_node

    def test_execution_trace_start_stop(self, device):
        use_device = (
            torch.profiler.ProfilerActivity.CUDA
            or torch.profiler.ProfilerActivity.XPU in supported_activities()
            or torch.profiler.ProfilerActivity.HPU in supported_activities()
        )
        # Create a temp file to save execution trace data.
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        expected_loop_events = 0
        et = ExecutionTraceObserver().register_callback(fp.name)
        for idx in range(10):
            if idx == 3:
                et.start()
            elif idx == 5:
                et.stop()
            elif idx == 8:
                et.start()
            elif idx == 9:
                et.stop()
            if et._execution_trace_running:
                expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)

        assert fp.name == et.get_output_file_path()
        et.unregister_callback()
        nodes = self.get_execution_trace_root(fp.name)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        assert found_root_node
        assert loop_count == expected_loop_events

    def test_execution_trace_repeat_in_loop(self, device):
        use_device = (
            torch.profiler.ProfilerActivity.CUDA
            or torch.profiler.ProfilerActivity.XPU in supported_activities()
            or torch.profiler.ProfilerActivity.HPU in supported_activities()
        )
        iter_list = {3, 4, 6, 8}
        expected_loop_events = len(iter_list)
        output_files = []
        for idx in range(10):
            if idx in iter_list:
                # Create a temp file to save execution trace data.
                fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
                fp.close()
                output_files.append(fp.name)
                et = ExecutionTraceObserver().register_callback(fp.name)
                et.start()
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)
            if idx in iter_list:
                et.stop()
                et.unregister_callback()

        event_count = 0
        for et_file in output_files:
            nodes = self.get_execution_trace_root(et_file)
            found_root_node = False
            for n in nodes:
                assert "name" in n
                if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                    assert n["id"] == 1
                    found_root_node = True
                if n["name"].startswith("## LOOP "):
                    event_count += 1
            assert found_root_node
        assert event_count == expected_loop_events

    def test_execution_trace_no_capture(self):
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        et = ExecutionTraceObserver().register_callback(fp.name)

        assert fp.name == et.get_output_file_path()
        et.unregister_callback()
        nodes = self.get_execution_trace_root(fp.name)
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
        assert found_root_node

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/124500")
    def test_execution_trace_nested_tensor(self):
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()

        observer = ExecutionTraceObserver().register_callback(fp.name)

        def fn(nt):
            return nt.sin().cos()

        with torch.profiler.profile(execution_trace_observer=observer) as prof:
            for i in range(3):
                values = torch.rand((8 + i, 4 + i))
                offsets = torch.tensor([0, 2, 4, 6, 8 + i])
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                fn(nt)

        nodes = self.get_execution_trace_root(fp.name)
        found_cos = False
        for n in nodes:
            assert "name" in n
            if "cos" in n["name"]:
                found_cos = True
        assert found_cos


devices = ["cpu", "cuda"]
if TEST_XPU:
    devices.append("xpu")
if TEST_HPU:
    devices.append("hpu")
instantiate_device_type_tests(
    TestExecutionTrace, globals(), allow_xpu="xpu" in devices, only_for=devices
)

if __name__ == "__main__":
    run_tests()
