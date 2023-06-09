# Owner(s): ["oncall: profiler"]
import collections
import gc
import io
import json
import os
import re
import tempfile
import textwrap
import threading
import unittest
from unittest.mock import patch
import weakref
from dataclasses import dataclass, field
from typing import List, Optional

import expecttest
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.autograd.profiler import KinetoStepTracker
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    ExecutionGraphObserver,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch._C._profiler import _TensorMetadata
from torch.profiler._pattern_matcher import (
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import skipCUDAVersionIn
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_WINDOWS,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_WITH_ASAN,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import pickle

from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):

    @skipCUDAVersionIn([(11, 5)])  # https://github.com/pytorch/pytorch/issues/69023
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA
        """
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for outer_idx in range(10):
            with _profile(use_cuda=True):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss)))
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(not (is_increasing and max_diff > 100 * 1024),
                        msg='memory usage is increasing, {}'.format(str(last_rss)))

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                x, = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # Only testing that emit_nvtx runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_nvtx(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_cudagraph_profiling_workaround(self):
        import subprocess

        # repro taken from #75504
        # Launch in a separate process to catch hanging/illegal memory errors
        # and to make sure CUPTI isn't already initialized.
        p = subprocess.check_call([sys.executable, "-c", """
import os
import torch
from torch.profiler import ProfilerActivity, profile

def add_one(in_: torch.Tensor):
    return in_ + 1

sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)

# add this before cuda graphs are created
torch.profiler._utils._init_for_cuda_graphs()

add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))
zeros = torch.zeros(10, device="cuda")
out = add_one_graphed(zeros)
assert out[0] == 1

with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)
"""], universal_newlines=True, timeout=60)

        # ^ this will throw an exception if the script fails.


@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestProfilerITT(TestCase):

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                x, = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # Only testing that emit_itt runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_itt(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            with record_function("## TEST 1 ##", "1, 2, 3"):
                rf_handle = _record_function_with_args_enter("## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u)
                _record_function_with_args_exit(rf_handle)
            with record_function("## TEST 3 ##"):
                rf_handle = _record_function_with_args_enter("## TEST 4 ##")
                _record_function_with_args_exit(rf_handle)
        return prof

    def test_record_function(self):
        prof_result = self._record_function_with_param()
        found_test_1 = False
        found_test_2 = False
        found_test_3 = False
        found_test_4 = False
        for e in prof_result.function_events:
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                self.assertTrue(e.input_shapes == [[]])
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
            elif "## TEST 3 ##" == e.name:
                found_test_3 = True
                self.assertTrue(e.input_shapes == [])
            elif "## TEST 4 ##" == e.name:
                found_test_4 = True
                self.assertTrue(e.input_shapes == [])
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)
        self.assertTrue(found_test_3)
        self.assertTrue(found_test_4)

    def test_datapipe_with_record_function(self):
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            input_dp1 = dp.iter.IterableWrapper(range(4))
            input_dp2 = dp.iter.IterableWrapper(range(4, 8))
            input_dp3 = dp.iter.IterableWrapper(range(8, 12))
            output_dp = input_dp1.mux(input_dp2, input_dp3)
            output = list(output_dp)

        has_iter = False
        has_mux = False
        for e in prof.function_events:
            if has_iter and has_mux:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_mux and "Multiplexer" in e.name:
                has_mux = True
        self.assertTrue(has_iter)
        self.assertTrue(has_mux)

    def test_datapipe_delegation_with_profiler(self):
        class IDPIterator(torch.utils.data.IterDataPipe):
            def __init__(self):
                self.data = list(range(10))
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= 10:
                    self._idx = 0
                    raise StopIteration
                self._idx += 1
                return self.data[self._idx - 1]

            def get_value(self, idx):
                return self.data[idx]

        dp1 = IDPIterator()  # The object itself is an iterator
        self.assertEqual(5, dp1.get_value(5))
        it_dp1 = iter(dp1)  # This creates the 1st iterator
        self.assertEqual(5, it_dp1.get_value(5))  # type: ignore[attr-defined]
        self.assertEqual(list(range(10)), list(it_dp1))

        class IDPDelegator(torch.utils.data.IterDataPipe):
            def __init__(self, datapipe):
                self.datapipe = datapipe

            def __iter__(self):
                return iter(self.datapipe)

        dp2 = IDPDelegator(dp1)
        it_dp2 = iter(dp2)
        self.assertEqual(5, it_dp2.get_value(5))
        self.assertEqual(list(range(10)), list(it_dp2))

    def test_datapipe_with_record_function_fork(self):
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            input_dp = dp.iter.IterableWrapper(range(10))
            dp1, dp2, dp3 = input_dp.fork(num_instances=3)
            output1 = list(dp1)
        has_iter = False
        has_child = False
        for e in prof.function_events:
            if has_iter and has_child:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_child and "_ChildDataPipe" in e.name:
                has_child = True
        self.assertTrue(has_iter)
        self.assertTrue(has_child)


class TestExecutionGraph(TestCase):
    def payload(self, use_cuda=False):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with record_function("## TEST 1 ##", "1, 2, 3"):
            inf_val = float("inf")
            neg_inf_val = float("-inf")
            nan_val = float("nan")
            rf_handle = _record_function_with_args_enter("## TEST 2 ##", 1, False, 2.5, [u, u], (u, u),
                                                         "hello", u, inf_val, neg_inf_val, nan_val)
            x = torch.randn(10, 10, requires_grad=True)
            if use_cuda:
                x = x.cuda()
            y = torch.randn(10, 10, requires_grad=True)
            if use_cuda:
                y = y.cuda()
            z = x + y + x * y + x * y
            z.backward(z)
            gelu = nn.GELU()
            m = torch.randn(2)
            _ = gelu(m)
            if use_cuda:
                z = z.cpu()
            _record_function_with_args_exit(rf_handle)

    def get_execution_graph_root(self, output_file_name):
        nodes = []
        with open(output_file_name, 'r') as f:
            eg_graph = json.load(f)
            assert "nodes" in eg_graph
            nodes = eg_graph["nodes"]
        return nodes

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_execution_graph_with_kineto(self):
        trace_called_num = 0

        def trace_handler(p):
            nonlocal trace_called_num
            trace_called_num += 1

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        # Create a temp file to save execution graph data.
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
        fp.close()
        expected_loop_events = 0
        eg = ExecutionGraphObserver()
        eg.register_callback(fp.name)
        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                skip_first=3,
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler,
        ) as p:
            eg.start()
            for idx in range(10):
                expected_loop_events += 1
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(use_cuda=use_cuda)
                p.step()
            eg.stop()

        assert trace_called_num == 2
        assert fp.name == eg.get_output_file_path()

        # cleanup
        eg.unregister_callback()
        nodes = self.get_execution_graph_root(fp.name)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_graph|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        assert found_root_node
        assert loop_count == expected_loop_events

    def test_execution_graph_alone(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        # Create a temp file to save execution graph data.
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
        fp.close()
        expected_loop_events = 0

        eg = ExecutionGraphObserver()
        eg.register_callback(fp.name)
        eg.start()
        for idx in range(5):
            expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)
        eg.stop()

        assert fp.name == eg.get_output_file_path()
        eg.unregister_callback()
        nodes = self.get_execution_graph_root(fp.name)
        loop_count = 0
        # Expected tensor object tuple size, in th form of:
        # [tensor_id, storage_id, offset, numel, itemsize, device_str]
        tensor_tuple_size = 6
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_graph|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
            # Check if tensor tuple representation size is correct.
            if n["name"] == "## TEST 2 ##":
                assert len(n["inputs"][3][0]) == tensor_tuple_size
        assert found_root_node
        assert loop_count == expected_loop_events

    def test_execution_graph_start_stop(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        # Create a temp file to save execution graph data.
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
        fp.close()
        expected_loop_events = 0
        eg = ExecutionGraphObserver()
        eg.register_callback(fp.name)
        for idx in range(10):
            if idx == 3:
                eg.start()
            elif idx == 5:
                eg.stop()
            elif idx == 8:
                eg.start()
            elif idx == 9:
                eg.stop()
            if eg._execution_graph_running:
                expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)

        assert fp.name == eg.get_output_file_path()
        eg.unregister_callback()
        nodes = self.get_execution_graph_root(fp.name)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_graph|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        assert found_root_node
        assert loop_count == expected_loop_events

    def test_execution_graph_repeat_in_loop(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        iter_list = {3, 4, 6, 8}
        expected_loop_events = len(iter_list)
        output_files = []
        for idx in range(10):
            if idx in iter_list:
                # Create a temp file to save execution graph data.
                fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
                fp.close()
                output_files.append(fp.name)
                eg = ExecutionGraphObserver()
                eg.register_callback(fp.name)
                eg.start()
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)
            if idx in iter_list:
                eg.stop()
                eg.unregister_callback()

        event_count = 0
        for eg_file in output_files:
            nodes = self.get_execution_graph_root(eg_file)
            found_root_node = False
            for n in nodes:
                assert "name" in n
                if "[pytorch|profiler|execution_graph|process]" in n["name"]:
                    assert n["id"] == 1
                    found_root_node = True
                if n["name"].startswith("## LOOP "):
                    event_count += 1
            assert found_root_node
        assert event_count == expected_loop_events

    def test_execution_graph_no_capture(self):
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
        fp.close()
        eg = ExecutionGraphObserver()
        eg.register_callback(fp.name)

        assert fp.name == eg.get_output_file_path()
        eg.unregister_callback()
        nodes = self.get_execution_graph_root(fp.name)
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_graph|process]" in n["name"]:
                found_root_node = True
        assert found_root_node


@instantiate_parametrized_tests
class TestProfiler(TestCase):

    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite.")
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode
        """
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            return mod(x)

        with _profile(with_stack=True, use_kineto=kineto_available(), experimental_config=_ExperimentalConfig(verbose=True)) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = call_module(a)
            c = b.sum()
            c.backward()

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any("test_profiler" in entry for entry in e.stack))
                self.assertTrue(any((
                    "test_source" in entry or
                    "ts_method_1" in entry or
                    "ts_method_2" in entry) for entry in e.stack))

        # TODO: https://github.com/pytorch/kineto/issues/617
        if kineto_available() and not IS_WINDOWS:
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with io.open(fname, 'r') as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    self.assertEqual(len(matches), 1, repr([e["name"] for e in matches]))
                    return matches[0]

                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                self.assertEqual(module_event["args"]["Python parent id"], wrapper_event["args"]["Python id"])

        torch._C._set_graph_executor_optimize(prev_opt)

    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            "multiple_preexisting": ((False, False), ) * 2,
            "open_in_scope": ((True, False),),
            "close_in_scope": ((False, True),),
            "complex": (
                # Large number of background threads
                (False, False),
                (False, False),
                (False, False),
                (False, False),

                # some of which finish during profiling
                (False, True),
                (False, True),

                # And the profiled section is also multithreaded
                (True, False),
                (True, True),

            ),
        }.items(),
        name_fn=lambda name, thread_spec: name
    )
    @parametrize("work_in_main_thread", [True, False])
    def test_source_multithreaded(self, name, thread_spec, work_in_main_thread):
        """Test various threading configurations.

        `thread_spec` is a Tuple[Tuple[bool, bool], ...] where each pair is a
        thread. The first bool indicates if the thread should be started under
        the profiler context and the second is if it should be joined under the
        profiler context.
        """

        timeout = 15
        num_threads = len(thread_spec) + 1  # Main thread
        start_barrier = threading.Barrier(num_threads, timeout=timeout)
        end_barrier = threading.Barrier(num_threads, timeout=timeout)

        class Task(threading.Thread):

            def __init__(self):
                self._end_gate = threading.Event()
                super().__init__(daemon=True)
                self.start()
                self.finished = False

            def run(self):
                self._run(self._end_gate)

            def release(self):
                self._end_gate.set()

            @staticmethod
            def _run(end_gate=None):

                def known_preexisting_function():
                    start_barrier.wait()

                # Fixed point that we can use to test capture of functions
                # which are already running when profiling is enabled.
                known_preexisting_function()

                model = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )

                def invoked_during_run():
                    pass

                invoked_during_run()

                _ = model(torch.rand(4, 10))
                end_barrier.wait()

                if end_gate is not None:
                    end_gate.wait(timeout=timeout)

        threads = {}

        def add_threads(context: bool):
            for idx, (start_under_profiler, _) in enumerate(thread_spec):
                if start_under_profiler == context:
                    assert idx not in threads
                    threads[idx] = Task()

        def join_threads(context: bool):
            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                if end_under_profiler == context:
                    threads[idx].release()

            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                t = threads[idx]
                if end_under_profiler == context:
                    t.join(timeout=timeout)

        try:
            add_threads(False)
            with torch.profiler.profile(with_stack=True) as prof:
                # Threads added while the profiler are running will not be observed
                # since there is no way to hook into Python's thread start call to
                # register the observer. These are here purely to verify safety.
                add_threads(True)

                if work_in_main_thread:
                    Task._run()
                else:
                    start_barrier.wait()
                    end_barrier.wait()

                join_threads(True)
            join_threads(False)

        finally:
            # It is very important that we clean up everything because the
            # Python tracer will detect ALL active threads. (Even orphans from
            # prior failed tests.) If we don't clean up properly we can
            # contaminate subsequent tests.
            start_barrier.abort()
            end_barrier.abort()
            for t in threads.values():
                t.release()

            for t in threads.values():
                t.join(timeout=timeout)

            for t in threads.values():
                self.assertFalse(t.is_alive())

        roots = prof.profiler.kineto_results.experimental_event_tree()
        nodes = [node for node in _utils.traverse_dfs(roots) if isinstance(node.extra_fields, _ExtraFields_PyCall)]
        tid_counts = collections.Counter([node.start_tid for node in nodes])

        prior_threads = sum(not start_under_profiler for start_under_profiler, _ in thread_spec)
        expected_threads = prior_threads + 1
        self.assertEqual(len(tid_counts), expected_threads, f"{expected_threads}, {tid_counts}")
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2 ** 64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [node.start_tid for node in nodes if "known_preexisting_function" in node.name]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [node.start_tid for node in nodes if "invoked_during_run" in node.name]
        self.assertEqual(len(observed_during_run), worker_threads)
        self.assertEqual(len(observed_during_run), len(set(observed_during_run)))

    def payload(self, use_cuda=False):
        x = torch.randn(10, 10)
        if use_cuda:
            x = x.cuda()
        y = torch.randn(10, 10)
        if use_cuda:
            y = y.cuda()
        z = torch.mm(x, y)
        z = z + y
        if use_cuda:
            z = z.cpu()

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        # rerun to avoid initial start overhead
        with _profile(use_cuda=use_cuda, use_kineto=True) as p:
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
        # print(output)
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name or "Cijk" in e.name:
                found_gemm = True
            if "Memcpy" in e.name or "memcpy" in e.name:
                found_memcpy = True
        if use_cuda:
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_kineto_multigpu(self):
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA]) as prof:
            for gpu_id in [0, 1]:
                x = torch.randn(10, 10).cuda(gpu_id)
                y = torch.randn(10, 10).cuda(gpu_id)
                z = x.matmul(y)

        found_gemm_0 = False
        found_gemm_1 = False
        found_cuda = False
        for evt in prof.events():
            if "gemm" in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                if evt.device_index == 0:
                    found_gemm_0 = True
                elif evt.device_index == 1:
                    found_gemm_1 = True
            if "cuda" in evt.name.lower() and evt.device_type == DeviceType.CPU:
                found_cuda = True

        self.assertTrue(found_gemm_0)
        self.assertTrue(found_gemm_1)
        self.assertTrue(found_cuda)

    def test_memory_profiler(self):
        def run_profiler(tensor_creation_fn):
            # collecting allocs / deallocs
            with _profile(profile_memory=True, record_shapes=True, use_kineto=kineto_available()) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[alloc_fn] > 0)
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[dealloc_fn] < 0)

        def create_cpu_tensor():
            return torch.rand(10, 10)

        def create_cuda_tensor():
            return torch.rand(10, 10).cuda()

        def create_mkldnn_tensor():
            return torch.rand(10, 10, dtype=torch.float32).to_mkldnn()

        stats = run_profiler(create_cpu_tensor)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "test_user_scope_alloc",
            ],
            deallocs=[
                "test_user_scope_dealloc",
            ]
        )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                with profile(profile_memory=True) as prof:
                    x = None
                    with record_function("test_user_scope_alloc"):
                        x = create_cpu_tensor()
                    with record_function("test_user_scope_dealloc"):
                        del x
                prof.export_chrome_trace(fname)
                with io.open(fname, 'r') as f:
                    trace = json.load(f)
                    assert "traceEvents" in trace
                    events = trace["traceEvents"]
                    found_memory_events = False
                    for evt in events:
                        assert "name" in evt
                        if evt["name"] == "[memory]":
                            found_memory_events = True
                            assert "args" in evt
                            assert "Addr" in evt["args"]
                            assert "Device Type" in evt["args"]
                            assert "Device Id" in evt["args"]
                            assert "Bytes" in evt["args"]

                            # Memory should be an instantaneous event.
                            assert "dur" not in evt["args"]
                            assert "cat" not in evt["args"]
                    assert found_memory_events

        if torch.cuda.is_available():
            create_cuda_tensor()
            stats = run_profiler(create_cuda_tensor)
            check_metrics(
                stats,
                "cuda_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ]
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ]
            )

        if torch._C.has_mkldnn:
            create_mkldnn_tensor()
            stats = run_profiler(create_mkldnn_tensor)
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::rand",
                    "aten::empty",
                    "aten::to_mkldnn",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ]
            )

        # check top-level memory events
        with _profile(profile_memory=True, use_kineto=kineto_available()) as prof:
            x = torch.rand(10, 10)
            del x
            if torch.cuda.is_available():
                y = torch.rand(10, 10).cuda()
                del y
            gc.collect()
        stats = prof.key_averages(group_by_input_shape=True)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::rand",
                "aten::empty"
            ],
            deallocs=[
                "[memory]"
            ]
        )
        if torch.cuda.is_available():
            check_metrics(
                stats,
                "cuda_memory_usage",
                deallocs=[
                    "[memory]"
                ]
            )

    @unittest.skipIf(IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared")
    def test_oom_tracing(self):
        def run_profiler(tensor_creation_fn):
            with _profile(profile_memory=True, record_shapes=True) as prof:
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        def create_cuda_tensor_oom():
            device = torch.device("cuda:0")
            return torch.empty(1024, 1024, 1024, 20, dtype=torch.float32, device=device)

        def check_trace(fname):
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                trace = json.load(f)
                self.assertTrue("traceEvents" in trace)
                events = trace["traceEvents"]
                found_out_of_memory_events = False
                for evt in events:
                    self.assertTrue("name" in evt)
                    if evt["name"] == "[OutOfMemory]":
                        found_out_of_memory_events = True
                        self.assertTrue("args" in evt)
                        self.assertTrue("Device Type" in evt["args"])
                        self.assertTrue("Device Id" in evt["args"])
                        self.assertTrue("Bytes" in evt["args"])

                        # Memory should be an instantaneous event.
                        self.assertTrue("dur" not in evt["args"])
                        self.assertTrue("cat" not in evt["args"])
                self.assertTrue(found_out_of_memory_events)

        if torch.cuda.is_available():
            with TemporaryFileName(mode="w+") as fname:
                prof = run_profiler(create_cuda_tensor_oom)
                check_trace(fname)




    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_module_hierarchy(self):
        class A(nn.Module):
            def my_new_method(self, x):
                return x * 3

            def forward_impl_(self, x, y):
                return self.my_new_method(x) + y

            def forward(self, x, y):
                y = y - 2
                return self.forward_impl_(x, y)

        class B(nn.Module):
            def forward(self, x):
                return x + 2

        class C(nn.Module):
            def __init__(self):
                super().__init__()
                self.A0 = A()
                self.B0 = B()

            def call_b(self, x):
                return self.B0.forward(x)

            def forward(self, x, y):
                return self.A0.forward(x, y) + self.call_b(x)

        model = C()
        model = torch.jit.script(model)
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)
        op_to_module_hierarchy = {}
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.", "TOP(C)::forward."]
        with TemporaryFileName(mode="w+") as fname:
            with profile(activities=[torch.profiler.ProfilerActivity.CPU], with_modules=True,) as prof:
                model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                trace = json.load(f)
                assert "traceEvents" in trace
                events = trace["traceEvents"]
                found_memory_events = False
                for evt in events:
                    assert "name" in evt
                    if "args" in evt:
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            hierarchy = evt["args"]["Module Hierarchy"]
                            if op_name in op_to_module_hierarchy:
                                assert hierarchy in op_to_module_hierarchy[op_name]

    def test_high_level_trace(self):
        """Checks that python side high level events are recorded.
        """
        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        def train():
            for _, data in enumerate(dataloader):
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count.keys():
                        actual_event_count[key] = actual_event_count.setdefault(key, 0) + 1
            for key, count in expected_event_count.items():
                self.assertTrue((key in actual_event_count.keys()) and (count == actual_event_count[key]))

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N
        }
        judge(expected_event_count, prof)

    def test_flops(self):
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        with _profile(record_shapes=True, with_flops=True, use_kineto=kineto_available()) as prof:
            model(inputs)
        profiler_output = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)
        self.assertIn("Total MFLOPs", profiler_output)
        if not (kineto_available() and torch.cuda.is_available()):
            return

        with profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1)
        self.assertIn("Total MFLOPs", profiler_output)

    def test_kineto_profiler_api(self):
        called_num = [0]

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        initial_step = KinetoStepTracker.current_step()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
        ) as p:
            for idx in range(8):
                self.payload(use_cuda=use_cuda)
                p.step()

        self.assertEqual(called_num[0], 2)
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        # case without schedule
        with profile(
            activities=supported_activities()
        ) as p:
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
        # print(output)

        test_schedule = torch.profiler.schedule(
            skip_first=2,
            wait=1,
            warmup=1,
            active=2,
            repeat=2)
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    def test_kineto_profiler_multiple_steppers(self):
        niters = 8
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        net = SimpleNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        opt.zero_grad()
        inputs = torch.rand(10)

        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def optimizer_step():
            """This simulates a step() hook in the optimizer"""
            KinetoStepTracker.increment_step("yet_another_step")

        initial_step = KinetoStepTracker.current_step()

        def run_batch():
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
            # Manually call the hook. TODO: Remove this once we add the
            # profiler step hooks in the Optimizer class that will get triggered above.
            # See https://github.com/pytorch/pytorch/issues/88446
            optimizer_step()

        for idx in range(niters):
            run_batch()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
        ) as p:
            for idx in range(niters):
                run_batch()
                p.step()

        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 2 * niters)

    def test_export_stacks(self):
        with _profile(with_stack=True, use_kineto=kineto_available(), experimental_config=_ExperimentalConfig(verbose=True)) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with io.open(fname, 'r') as f:
                lines = f.readlines()
            assert len(lines) > 0, "Empty stacks file"
            for line in lines:
                is_int = False
                try:
                    assert int(line.split(" ")[-1]) > 0, "Invalid stacks record"
                    is_int = True
                except ValueError:
                    pass
                assert is_int, "Invalid stacks record"

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_tensorboard_trace_handler(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU
                ] + ([
                    torch.profiler.ProfilerActivity.CUDA
                ] if use_cuda else []),
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname)
            ) as p:
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split('.')
                self.assertTrue(len(parts) > 4)
                self.assertTrue(parts[-4].isdigit() and int(parts[-4]) > 0, "Wrong tracing file name pattern")
                self.assertEqual(parts[-3:], ['pt', 'trace', 'json'])
                file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            p = profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU
                ] + ([
                    torch.profiler.ProfilerActivity.CUDA
                ] if use_cuda else []),
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname, use_gzip=True)
            )
            p.start()
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            p.stop()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split('.')
                self.assertTrue(len(parts) > 4)
                self.assertTrue(parts[-5].isdigit() and int(parts[-5]) > 0, "Wrong tracing file name pattern")
                self.assertEqual(parts[-4:], ['pt', 'trace', 'json', 'gz'])
                file_num += 1
            self.assertEqual(file_num, 3)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_metadata(self):
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                trace = json.load(f)
                assert "test_key1" in trace
                assert trace["test_key1"] == "test_value1"
                assert "test_key2" in trace
                assert trace["test_key2"] == [1, 2, 3]

    def _test_profiler_tracing(self, use_kineto):
        with _profile(use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1), torch.ones(1)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # read the trace and expect valid json
            # if the JSON generated by export_chrome_trace is not valid, this will throw and fail the test.
            with io.open(fname, 'r') as f:
                json.load(f)

        # test empty trace
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # saving an empty trace
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)

        # Same test but for cuda.
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        device = torch.device("cuda:0")
        with _profile(use_cuda=True, use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # Now validate the json
            with io.open(fname, 'r') as f:
                json.load(f)

    def test_profiler_tracing(self):
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    def test_profiler_fwd_bwd_link(self):
        with _profile(use_kineto=True) as prof:
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                j = json.load(f)
                events = j["traceEvents"]
                ts_to_name = {}
                flow_s_to_ts = {}
                flow_f_to_ts = {}
                for e in events:
                    if e["ph"] == "X":
                        ts_to_name[e["ts"]] = e["name"]
                    if "cat" in e and "name" in e and e["cat"] == "fwdbwd" and e["name"] == "fwdbwd":
                        if e["ph"] == "s":
                            flow_s_to_ts[e["id"]] = e["ts"]
                        elif e["ph"] == "f":
                            flow_f_to_ts[e["id"]] = e["ts"]

                self.assertEqual(len(flow_s_to_ts), 2)
                self.assertEqual(len(flow_f_to_ts), 2)
                self.assertIn(1, flow_s_to_ts)
                self.assertIn(1, flow_f_to_ts)
                self.assertIn(2, flow_s_to_ts)
                self.assertIn(2, flow_f_to_ts)
                s_ts_1 = flow_s_to_ts[1]
                f_ts_1 = flow_f_to_ts[1]
                s_ts_2 = flow_s_to_ts[2]
                f_ts_2 = flow_f_to_ts[2]
                self.assertTrue(all(ts in ts_to_name.keys() for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2]))
                self.assertTrue(ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits")
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")

    def test_profiler_disable_fwd_bwd_link(self):
        try:
            torch._C._profiler._set_fwd_bwd_enabled_val(False)

            with _profile(use_kineto=True) as prof:
                t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                z = torch.add(t1, t2)
                y = torch.ones(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                loss.backward()

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with io.open(fname, 'r') as f:
                    j = json.load(f)
                    events = j["traceEvents"]

                    for e in events:
                        self.assertNotEqual(getattr(e, "cat", None), "fwdbwd")
        finally:
            torch._C._profiler._set_fwd_bwd_enabled_val(True)


    def test_profiler_type(self):
        profiler_type = torch._C._autograd._profiler_type
        ActiveProfilerType = torch._C._profiler.ActiveProfilerType
        self.assertEqual(profiler_type(), ActiveProfilerType.NONE)

        # Autograd profiler
        with _profile_legacy():
            self.assertEqual(profiler_type(), ActiveProfilerType.LEGACY)

        # Kineto profiler
        with profile():
            self.assertEqual(profiler_type(), ActiveProfilerType.KINETO)

    def test_profiler_correlation_id(self):
        '''
        We expect the correlation_id to be unique across multiple invokation of the profiler,
        So we will reuse id_uniqueness_set.
        '''
        id_uniqueness_set = set()
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        uint32_max = 2**32 - 1
        for i in range(5):
            with profile() as prof:
                model(inputs)
            for event in prof.profiler.kineto_results.events():
                corr_id = event.correlation_id()
                if (corr_id):
                    self.assertTrue(corr_id not in id_uniqueness_set)
                    id_uniqueness_set.add(corr_id)
                    self.assertTrue(corr_id < uint32_max)

    def test_nested_tensor_with_shapes(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        inp = torch.nested.nested_tensor([a, b])
        with torch.profiler.profile(record_shapes=True) as prof:
            torch.nn.functional.linear(inp, c, None)
        for e in prof.events():
            if e.name in ("aten::mm", "aten::addmm"):
                # intentionally vague tests to protect against possible future changes
                # of mm to addmm or other impl, or changing internal order of args
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.input_shapes[0]) > 0)

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    def test_kineto_profiler_with_environment_variable(self):
        script = """
import torch
import torch.nn as nn
from torch.profiler import supported_activities, profile
from torch.autograd.profiler import KinetoStepTracker

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def payload(use_cuda=False):
    x = torch.randn(10, 10)
    if use_cuda:
        x = x.cuda()
    y = torch.randn(10, 10)
    if use_cuda:
        y = y.cuda()
    z = torch.mm(x, y)
    z = z + y
    if use_cuda:
        z = z.cpu()

niters = 8
use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
net = SimpleNet()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
opt.zero_grad()
inputs = torch.rand(10)

with profile(activities=supported_activities()):
    payload(use_cuda=use_cuda)

initial_step = KinetoStepTracker.current_step()

def run_batch():
    out = net(inputs)
    loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
    loss.backward()
    opt.step()

for _ in range(niters):
    run_batch()

with profile(
    activities=supported_activities(),
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
) as p:
    for _ in range(niters):
        run_batch()
        p.step()
assert KinetoStepTracker.current_step() == initial_step + 2 * niters
"""
        try:
            subprocess.check_output(
                [sys.executable, '-W', 'all', '-c', script],
                cwd=os.path.dirname(os.path.realpath(__file__))
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(False, "Kineto is not working properly with the Dynolog environment variable")

    def test_concrete_inputs_profiling(self):
        x = torch.rand(2, 6)
        with profile(record_shapes=True) as p:
            y = x.as_strided([4, 3], [1, 4])

        found = False
        for e in p.events():
            if e.name in ("aten::as_strided"):
                found = True
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.concrete_inputs) > 0)
                self.assertEqual([2, 6], e.input_shapes[0])
                self.assertEqual([4, 3], e.concrete_inputs[1])
                self.assertEqual([1, 4], e.concrete_inputs[2])

        self.assertTrue(found, "Expected to find aten::as_strided but did not")

    def test_concrete_inputs_profiling_toggling(self):
        try:
            for (before, after) in [(True, False), (False, True)]:
                x = torch.rand(2, 6)
                torch._C._profiler._set_record_concrete_inputs_enabled_val(before)
                with profile(record_shapes=True) as p:
                    y = x.as_strided([4, 3], [1, 4])
                    torch._C._profiler._set_record_concrete_inputs_enabled_val(after)

                found = False
                for e in p.events():
                    if e.name in ("aten::as_strided"):
                        found = True
                        self.assertTrue(len(e.input_shapes))

                self.assertTrue(found, "Expected to find aten::as_strided but did not")
        finally:
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)


def find_node_with_name(nodes, name):
    for node in _utils.traverse_dfs(nodes):
        if node.name == name:
            return node

def find_node_with_regex(nodes, pattern):
    for node in _utils.traverse_dfs(nodes):
        if re.search(pattern, node.name):
            return node


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestTorchTidyProfiler(TestCase):

    def _get_tensor_fields(self, node, index):
        self.assertIsNotNone(node)
        self.assertIsInstance(
            node.extra_fields,
            torch._C._profiler._ExtraFields_TorchOp)
        tensor_info = node.extra_fields.inputs[index]
        self.assertIsInstance(tensor_info, _TensorMetadata)
        self.assertIsNotNone(tensor_info.impl_ptr)
        self.assertIsNotNone(tensor_info.storage_data_ptr)
        self.assertIsNotNone(tensor_info.id)
        return tensor_info.impl_ptr, tensor_info.storage_data_ptr, tensor_info.id

    def test_pointers_and_ids(self):
        a = torch.randn(4, 3)
        a_initial_storage_data = a.storage().data_ptr()

        # Views of tensors can share the same storage, but have different TensorImpls
        b = a.view((1, 12))
        c = torch.randn(4, 1)
        c_initial_storage_data = c.storage().data_ptr()
        d = torch.randn(4, 3)

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = a + c
            _ = b * c

            # Resize should create a new data_ptr but keep the TensorImpl the same.
            f = a.resize_(128, 129)
            _ = torch.relu(f)

            # `.set_` points a Tensor at an existing storage.
            _ = d.sin()
            c.set_(d.storage())
            _ = c.cos()

        nodes = p.profiler.kineto_results.experimental_event_tree()

        def get_fields(op_name, index):
            return self._get_tensor_fields(
                find_node_with_name(nodes, op_name),
                index)

        a_impl, a_storage_data, a_id = get_fields("aten::add", 0)
        b_impl, b_storage_data, b_id = get_fields("aten::mul", 0)

        # Profiler matches ground truth from Python API.
        self.assertEqual(a_storage_data, a_initial_storage_data)

        # Views are handled correctly.
        self.assertEqual(a_storage_data, b_storage_data)
        self.assertNotEqual(a_impl, b_impl)

        # The same Tensor used in multiple calls gives identical results.
        c_impl, c_storage_data, c_id = get_fields("aten::add", 1)
        self.assertEqual((c_impl, c_storage_data, c_id), get_fields("aten::mul", 1))
        self.assertEqual(c_storage_data, c_initial_storage_data)

        # Mutations to the underlying storage are reflected. (But ID is shared.)
        f_impl, f_storage_data, f_id = get_fields("aten::relu", 0)
        self.assertEqual(a_impl, f_impl)
        self.assertNotEqual(a_storage_data, f_storage_data)
        self.assertEqual(a_id, f_id)

        # Calling `set_` with an existing Tensor makes them share an ID.
        d_impl, d_storage_data, d_id = get_fields("aten::sin", 0)
        c_impl_new, c_storage_data_new, c_id_new = get_fields("aten::cos", 0)
        self.assertNotEqual(d_impl, c_impl_new)
        self.assertEqual(d_storage_data, c_storage_data_new)
        self.assertEqual(c_id, c_id_new)
        self.assertEqual(d_id, c_id_new)

    @staticmethod
    def _format_allocations(profiled_code):
        gc.collect()
        with profile(profile_memory=True, record_shapes=True) as prof:
            profiled_code()
            gc.collect()

        root_events = prof.profiler.kineto_results.experimental_event_tree()
        events = sorted(_utils.traverse_dfs(root_events), key=lambda x: x.start_time_ns)
        allocations = tuple(
            event.extra_fields
            for event in events
            if isinstance(event.extra_fields, torch._C._profiler._ExtraFields_Allocation)
        )

        return textwrap.indent("\n".join(
            f"{repr(i.id):>5}{' ' * 6}"
            f"{repr(i.allocation_id):>5}{' ' * 6}"
            f"{'Allocation' if i.alloc_size > 0 else 'Free'}"
            for i in allocations
        ), " " * 12)

    def test_tensorimpl_invalidation_set(self) -> None:
        def profiled_code(add_empty_set: bool):
            x = torch.ones((1,))

            # Determines if new storage is created before or after the old one
            # is destroyed.
            if add_empty_set:
                x.set_()

            x.set_(torch.ones((1,)).storage())
            x.view_as(x)

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          1      Free
                0          2      Free"""
        )

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          1      Free
                0          2      Allocation
                0          2      Free"""
        )

    def test_tensorimpl_invalidation_keep_alive(self) -> None:
        def profiled_code(add_empty_set: bool):
            x = torch.ones((1,))
            x_storages = [x.storage()]
            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())

                # This keeps the StorageImpls alive and preserves the chain.
                # (Despite the `set_()` call.)
                x_storages.append(x.storage())
            x.view_as(x)

            # Free storage in a deterministic fashion.
            while x_storages:
                x_storages.pop()
                gc.collect()

            # Determines if new storage is created before or after the old one
            # is destroyed.
            if add_empty_set:
                x.set_()

            for _ in range(3):
                x.set_(torch.ones((1,)).storage())
            x.view_as(x)

            del x
            gc.collect()

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free"""
        )

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          5      Free
                0          6      Allocation
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free"""
        )

    def test_tensorimpl_invalidation_full(self) -> None:
        def profiled_code():
            x = torch.ones((1,))
            x_storages = [x.storage()]
            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())
                x_storages.append(x.storage())
            x.view_as(x)

            # Free storage in a deterministic fashion.
            while x_storages:
                x_storages.pop()
                gc.collect()

            for _ in range(3):
                x.set_(torch.ones((1,)).storage())

            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())

            for i in range(4):
                x.resize_((1 + i,))
            x.view_as(x)

        self.assertExpectedInline(
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free
                0          9      Allocation
                0          9      Free
                0         10      Allocation
                0         10      Free
                0         11      Allocation
                0         12      Allocation
                0         11      Free
                0         13      Allocation
                0         12      Free
                0         14      Allocation
                0         13      Free
                0         14      Free"""
        )

    def test_tensorimpl_invalidation_scalar_args(self) -> None:
        def profiled_code():
            with torch.no_grad():
                x = torch.ones((1,))
                for _ in range(10):
                    x.add_(2)

        self.assertExpectedInline(
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                1          2      Allocation
                2          3      Allocation
                2          3      Free
                1          2      Free
                3          4      Allocation
                4          5      Allocation
                4          5      Free
                3          4      Free
                5          6      Allocation
                6          7      Allocation
                6          7      Free
                5          6      Free
                7          8      Allocation
                8          9      Allocation
                8          9      Free
                7          8      Free
                9         10      Allocation
               10         11      Allocation
               10         11      Free
                9         10      Free
               11         12      Allocation
               12         13      Allocation
               12         13      Free
               11         12      Free
               13         14      Allocation
               14         15      Allocation
               14         15      Free
               13         14      Free
               15         16      Allocation
               16         17      Allocation
               16         17      Free
               15         16      Free
               17         18      Allocation
               18         19      Allocation
               18         19      Free
               17         18      Free
               19         20      Allocation
               20         21      Allocation
               20         21      Free
               19         20      Free
                0          1      Free""")


    def test_module_and_optimizer_ids(self) -> None:
        model = torch.nn.Linear(2, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def check(cold_start: bool) -> None:
            with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
                x = torch.ones((1, 2))
                _ = x.sin()  # Mark `x`
                model(x).backward()
                optimizer.step()
                _ = optimizer.state[model.weight]["momentum_buffer"].cos()  # Mark weight momentum
                _ = model.weight.grad.tan()  # Mark weight gradient

            nodes = p.profiler.kineto_results.experimental_event_tree()

            def get_fields(op_name, index):
                return self._get_tensor_fields(
                    find_node_with_name(nodes, op_name),
                    index)

            # Marked Tensors act as ground truth for python tracer IDs.
            _, _, x_id = get_fields("aten::sin", 0)
            _, _, weight_momenumtum_id = get_fields("aten::cos", 0)
            _, _, weight_grad_id = get_fields("aten::tan", 0)
            self.assertNotEqual(x_id, weight_momenumtum_id)
            self.assertNotEqual(x_id, weight_grad_id)
            self.assertNotEqual(weight_momenumtum_id, weight_grad_id)

            # Use linear op to identify weight ground truth.
            linear_op_node = find_node_with_name(nodes, "aten::linear")
            self.assertIsNotNone(linear_op_node)
            x_metadata, weight_metadata, _ = linear_op_node.extra_fields.inputs
            self.assertEqual(x_id, x_metadata.id)

            # Module
            linear_module_node = find_node_with_name(nodes, "nn.Module: Linear_0")
            self.assertIsNotNone(linear_module_node)
            self.assertIsNotNone(linear_module_node.extra_fields.module)
            self.assertIsNone(linear_module_node.extra_fields.optimizer)

            linear_parameters = linear_module_node.extra_fields.module.parameters
            name, weight, weight_grad = linear_parameters[0]
            self.assertEqual(name, "weight")
            self.assertEqual(weight.id, weight_metadata.id)

            self.assertEqual(weight_grad is None, cold_start)
            if not cold_start:
                self.assertEqual(weight_grad.id, weight_grad_id)

            # Optimizer
            step_node = find_node_with_regex(nodes, "_optimizer_step_code")
            self.assertIsNotNone(step_node)
            self.assertIsNone(step_node.extra_fields.module)
            self.assertIsNotNone(step_node.extra_fields.optimizer)
            optimizer_parameters = step_node.extra_fields.optimizer.parameters
            self.assertEqual(len(optimizer_parameters), 2)  # Weight and bias
            weight, weight_grad, state = optimizer_parameters[0]
            self.assertEqual(weight.id, weight_metadata.id)
            self.assertEqual(weight_grad.id, weight_grad_id)
            self.assertEqual(len(state), 1)
            self.assertEqual(state[0][0], "momentum_buffer")
            self.assertEqual(state[0][1].id, weight_momenumtum_id)

        # Check that we handle first step (lazy initalization) and steady state.
        check(cold_start=True)
        check(cold_start=False)

    def _test_allocation_ids(self, before_fn, after_fn) -> None:
        with profile(profile_memory=True, record_shapes=True) as p:
            # Introduce other operations and allocations to check robustness
            _ = before_fn()

            x = torch.rand(4, 3)
            x.resize_(4, 4)

            # We need to use `x` post resize for profiler to determine its ID.
            x.sin()

            # Introduce other operations and allocations to check robustness
            _ = after_fn()

            # Ensure `x` is the last variable collected to make it easier to
            # find the deallocation event.
            gc.collect()
            del x
            gc.collect()

        nodes = p.profiler.kineto_results.experimental_event_tree()

        def find_chain(names: List[str]):
            out = []
            for name in names:
                root = [out[-1]] if out else nodes
                out.append(find_node_with_name(root, name))
                self.assertIsNotNone(out[-1], name)
            return out

        allocation = find_chain(["aten::rand", "aten::empty", "[memory]"])[-1].extra_fields
        _, uniform_node = find_chain(["aten::rand", "aten::uniform_"])
        x_impl, x_storage_data, x_id = self._get_tensor_fields(uniform_node, 0)

        # Make sure IDs are consistent between allocations and op inputs
        self.assertEqual(allocation.ptr, x_storage_data)
        self.assertEqual(allocation.id, x_id)

        resize_node = find_node_with_name(nodes, "aten::resize_")
        self.assertIsNotNone(resize_node)
        self.assertEqual(len(resize_node.children), 2)
        allocate_new = resize_node.children[0].extra_fields
        free_old = resize_node.children[1].extra_fields

        # Destruction of the old storage for x.
        self.assertEqual(free_old.id, allocation.id)
        self.assertEqual(free_old.ptr, allocation.ptr)

        # Make sure ID is retained through change in storage.
        self.assertEqual(allocate_new.id, allocation.id)
        self.assertNotEqual(allocate_new.ptr, allocation.ptr)

        # Deletion when `x` goes out of scope.
        free_new = [i for i in nodes if i.tag == torch._C._profiler._EventType.Allocation][-1].extra_fields
        self.assertIsInstance(free_new, torch._C._profiler._ExtraFields_Allocation)
        self.assertEqual(free_new.id, allocate_new.id)
        self.assertEqual(free_new.ptr, allocate_new.ptr)

    def test_allocation_ids(self) -> None:
        self._test_allocation_ids(lambda: None, lambda: None)

    def test_allocation_ids_with_other_ops(self) -> None:
        x = torch.ones((1,))
        self._test_allocation_ids(
            lambda: (x + 1).relu_(),
            lambda: torch.zeros((1,)).cos()
        )

    def test_impl_reuse(self) -> None:
        repeats = 1_000
        with profile(profile_memory=True, record_shapes=True) as p:
            for _ in range(repeats):
                torch.ones((1,))
            gc.collect()

        roots = p.profiler.kineto_results.experimental_event_tree()
        tensor_impls = tuple(
            e.extra_fields.inputs[0].impl_ptr
            for e in _utils.traverse_dfs(roots)
            if e.name == "aten::fill_"
        )

        self.assertEqual(len(tensor_impls), repeats)
        self.assertEqual(len(set(tensor_impls)), repeats)

    def test_allocation_id_uniqueness(self) -> None:
        repeats = 1_000
        with profile(profile_memory=True, record_shapes=True) as p:
            for _ in range(repeats):
                torch.ones((1,))
            gc.collect()

        roots = p.profiler.kineto_results.experimental_event_tree()
        id_set = set()
        for e in _utils.traverse_dfs(roots):
            fields = e.extra_fields
            if isinstance(fields, torch._C._profiler._ExtraFields_TorchOp):
                id_set |= {t.allocation_id for t in fields.inputs if isinstance(t, _TensorMetadata)}

            elif isinstance(fields, torch._C._profiler._ExtraFields_Allocation):
                id_set.add(fields.allocation_id)

        id_set.difference_update([None])
        self.assertEqual(repeats, len(id_set))

    def test_extra_fields(self):
        with profile(with_stack=True, profile_memory=True) as p:
            _ = torch.ones((1,))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::ones")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields,
            torch._C._profiler._ExtraFields_TorchOp)

        self.assertIsInstance(
            node.parent.extra_fields,
            torch._C._profiler._ExtraFields_PyCCall)

        self.assertEqual(node.children[0].name, "aten::empty")
        self.assertEqual(node.children[0].children[0].name, "[memory]")
        self.assertIsInstance(
            node.children[0].children[0].extra_fields,
            torch._C._profiler._ExtraFields_Allocation)

    def test_tensor_properties(self):
        x = torch.ones(10, 10).as_strided([4, 4], [12, 3])
        y = torch.ones(4, 1, requires_grad=True)

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = x + y
            _ = x * y

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields,
            torch._C._profiler._ExtraFields_TorchOp)

        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[4, 4], [4, 1], []])
        self.assertEqual(getattr_inputs("strides", []), [[12, 3], [1, 1], []])
        self.assertEqual(getattr_inputs("layout", None), [torch.strided, torch.strided, None])
        self.assertEqual(getattr_inputs("device", None), [torch.device("cpu"), torch.device("cpu"), None])
        self.assertEqual(getattr_inputs("dtype", None), [torch.float32, torch.float32, None])
        self.assertEqual(node.extra_fields.scope, torch.profiler.RecordScope.FUNCTION)

        mul_node = find_node_with_name(nodes, "aten::mul")
        self.assertIsNotNone(mul_node)
        self.assertEqual(
            node.extra_fields.sequence_number + 1,
            mul_node.extra_fields.sequence_number)

    def test_sparse_tensors(self):
        i = [[0, 1, 1], [2, 0, 2]]
        v = [3, 4, 5]
        s = torch.sparse_coo_tensor(i, v, (2, 3))

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = s + s

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields,
            torch._C._profiler._ExtraFields_TorchOp)

        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[2, 3], [2, 3], []])
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        self.assertEqual(getattr_inputs("layout", None), [torch.sparse_coo, torch.sparse_coo, None])
        self.assertEqual(getattr_inputs("device", None), [torch.device("cpu"), torch.device("cpu"), None])

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_mkldnn_tensors(self):
        x = torch.ones(4, 3).to_mkldnn()

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = x + x

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields,
            torch._C._profiler._ExtraFields_TorchOp)

        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[4, 3], [4, 3], []])
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        self.assertEqual(getattr_inputs("layout", None), [torch._mkldnn, torch._mkldnn, None])
        self.assertEqual(getattr_inputs("device", None), [torch.device("cpu"), torch.device("cpu"), None])

    def test_scalar_ins(self):
        x = torch.ones(5, 5)
        alpha = 0.9

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = torch.add(x, 9.1, alpha=alpha)

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # The second argument to the add gets promotoed to a zerodim Tensor
        self.assertEqual(getattr_inputs("dtype", None), [torch.float32, torch.float64, None])
        self.assertEqual(getattr_inputs("sizes", []), [[5, 5], [], []])
        self.assertEqual(node.extra_fields.inputs[2], alpha)

    def test_tensor_lists(self):
        x = torch.ones((1,))
        y = torch.ones((1,))
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = torch.stack((x, y))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::stack")
        inputs = node.extra_fields.inputs
        self.assertEqual(len(inputs), 2)
        self.assertIsInstance(inputs[0], list)
        self.assertEqual(len(inputs[0]), 2)
        self.assertEqual(x.storage().data_ptr(), inputs[0][0].storage_data_ptr)
        self.assertEqual(y.storage().data_ptr(), inputs[0][1].storage_data_ptr)


    def test_nnmodule_params(self):

        def flat_out_extrafields(nodes, out=None):
            if out is None:
                out = []
            for node in nodes:
                if isinstance(node.extra_fields, _ExtraFields_PyCall) and node.extra_fields.module:
                    if node.extra_fields.module.parameters:
                        out.append(node.extra_fields.module)
                flat_out_extrafields(node.children, out)
            return out

        inputs = torch.rand(10)
        net = SimpleNet()
        out = net(inputs)
        torch.nn.functional.cross_entropy(out, torch.rand(2)).backward()
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            _ = net(inputs)

        modules = flat_out_extrafields(p.profiler.kineto_results.experimental_event_tree())
        self.assertEqual(len(modules), 2, f"Expected two parameter list, but got {len(modules)}")

        params = [(n, p.storage_data_ptr, g.storage_data_ptr) for module in modules for (n, p, g) in module.parameters]
        expected = [(name, val.storage().data_ptr(), val.grad.storage().data_ptr()) for name, val in net.fc1._parameters.items()]
        expected += [(name, val.storage().data_ptr(), val.grad.storage().data_ptr()) for name, val in net.fc2._parameters.items()]
        self.assertEqual(expected, params, f"{expected} vs. {params}")

    def _flat_out_extrafields(self, nodes, out=None):
        if out is None:
            out = []
        for node in nodes:
            if (isinstance(node.extra_fields, _ExtraFields_PyCall) and
                    node.extra_fields.optimizer and node.extra_fields.optimizer.parameters):
                # avoiding OptInfo duplicates from iterations
                addr = node.extra_fields.optimizer.parameters[0][0].storage_data_ptr
                if not [o for o in out if addr == o.parameters[0][0].storage_data_ptr]:
                    out.append(node.extra_fields.optimizer)
            self._flat_out_extrafields(node.children, out)
        return out

    def _check_results(self, opt, opts, check_items=False):
        self.assertEqual(len(opts), 1, f"Expected 1 optimizer: len(opts): {len(opts)}")
        self.assertEqual(id(opt), opts[0].self_ptr, f"Optimizer addr ({id(opt)}) vs. profiled addr ({opts[0].self_ptr})")
        if check_items:
            self.assertEqual(len(opt.param_groups), len(opts))
            for group, opt_ in zip(opt.param_groups, opts):
                self.assertEqual(
                    [(v.storage().data_ptr()) for v in group.get("params", [])],
                    [(o.storage_data_ptr) for (o, _, _) in opt_.parameters]
                )
            for opt_ in opts:
                observed_state = {
                    p.storage_data_ptr: {name: s.storage_data_ptr for name, s in state}
                    for (p, _, state) in opt_.parameters
                }

                # Make sure the profiler collected all optimizer state and check
                # that the address recorded by the profiler is correct.
                for parameter, parameter_state in opt.state.items():
                    self.assertEqual(
                        {name: value.storage().data_ptr() for name, value in parameter_state.items()},
                        observed_state.get(parameter.storage().data_ptr(), [])
                    )

    def test_optimizer(self):
        inputs = torch.rand(10)
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            net = SimpleNet()
            opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

            opt.zero_grad()
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
        self._check_results(opt, self._flat_out_extrafields(p.profiler.kineto_results.experimental_event_tree()), False)

    def _test_optimizer_parameters(self, optimizer_factory):
        inputs = torch.rand(10)
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            net = SimpleNet()
            opt = optimizer_factory(net.parameters())
            for _ in range(2):
                opt.zero_grad()
                out = net(inputs)
                loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
                loss.backward()
                opt.step()
        self._check_results(opt, self._flat_out_extrafields(p.profiler.kineto_results.experimental_event_tree()), True)

    def test_optimizer_parameters_sgd(self):
        self._test_optimizer_parameters(lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9))

    def test_optimizer_parameters_adam(self):
        self._test_optimizer_parameters(lambda params: torch.optim.Adam(params, foreach=True))

    def test_allocations(self):
        gc.collect()
        with profile(profile_memory=True) as p:
            x = torch.empty((3, 4))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "[memory]")
        self.assertIsNotNone(node)

        alloc_size = 3 * 4 * 4  # fp32 -> 4 bytes
        ptr = node.extra_fields.ptr
        self.assertGreater(ptr, 0)
        self.assertEqual(node.extra_fields.alloc_size, alloc_size)
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        total_allocated = node.extra_fields.total_allocated

        # total_reserved is only for CUDACachingAllocator
        self.assertEqual(node.extra_fields.total_reserved, 0)

        with profile(profile_memory=True) as p:
            del x
            gc.collect()

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "[memory]")
        self.assertIsNotNone(node)

        self.assertEqual(node.extra_fields.ptr, ptr)
        self.assertEqual(node.extra_fields.alloc_size, -alloc_size)
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        self.assertEqual(node.extra_fields.total_allocated, total_allocated - alloc_size)

    def test_refcounts(self):

        class Sentinel:
            pass

        def make():
            outer_sentinel = Sentinel()

            def outer():
                # Python will only close over variables used in the function.
                _ = outer_sentinel
                inner_sentinel = Sentinel()

                def inner():
                    _ = inner_sentinel


                with profile(with_stack=True):
                    inner()

                return weakref.ref(inner_sentinel)

            return outer, weakref.ref(outer_sentinel)

        # Use a factory function to ensure the test scope never sees strong
        # references. `del` has strange semantics that interact with closures
        # at an AST level, so this is simpler.
        outer, outer_sentinel_ref = make()
        inner_sentinel_ref = outer()

        self.assertIsNone(inner_sentinel_ref())

        # `outer` holds the last reference via closure.
        self.assertIsNotNone(outer_sentinel_ref())

        del outer
        self.assertIsNone(outer_sentinel_ref())


@dataclass(frozen=True)
class MockKinetoEvent():
    _name: str
    _start_us: int
    _duration_us: int
    _linked_correlation_id: int
    _device_type: int

    @property
    def name(self) -> str:
        return self._name

    def start_us(self) -> int:
        return self._start_us

    def duration_us(self) -> int:
        return self._duration_us

    def linked_correlation_id(self) -> int:
        return self._linked_correlation_id

    def device_type(self) -> DeviceType:
        return DeviceType.CUDA if self._device_type == 1 else DeviceType.CPU


@dataclass(frozen=True)
class MockProfilerEvent():
    _name: str
    id: int
    start_time_ns: int
    duration_time_ns: int
    correlation_id: int = 0
    children: List["MockProfilerEvent"] = field(default_factory=list)
    parent: Optional["MockProfilerEvent"] = None

    @property
    def end_time_ns(self):
        return self.start_time_ns + self.duration_time_ns

    @property
    def name(self) -> str:
        return self._name

    def __post__init__(self, parent, children):
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "children", children)

class MockNode:
    def __init__(self, name, children) -> None:
        self.name = name
        self.children = [MockNode(name, i) for name, i in children.items()]


class TestExperimentalUtils(TestCase):

    def make_tree(self) -> List[MockNode]:
        tree = {
            "root_0": {
                "1": {
                    "2": {}
                },
                "3": {
                    "4": {},
                    "5": {},
                },
            },
            "root_1": {
                "6": {},
                "7": {},
                "8": {
                    "9": {
                        "10": {}
                    },
                },
            },
        }
        return [MockNode(name, i) for name, i in tree.items()]

    def test_dfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_dfs(self.make_tree())),
            "root_0 1 2 3 4 5 root_1 6 7 8 9 10")

    def test_bfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_bfs(self.make_tree())),
            "root_0 root_1 1 3 6 7 8 2 4 5 9 10")

    @staticmethod
    def generate_mock_profile():
        cuda_events = [
            MockKinetoEvent("cudaLaunchKernel", 400, 100, 1, 0),
            MockKinetoEvent("cudaLaunchKernel", 500, 100, 2, 0),
            MockKinetoEvent("cudaLaunchKernel", 600, 100, 3, 0),
            MockKinetoEvent("cudaLaunchKernel", 700, 100, 4, 0),
            MockKinetoEvent("cudaLaunchKernel", 800, 100, 5, 0),
            MockKinetoEvent("cudaLaunchKernel", 1500, 100, 6, 0),
            MockKinetoEvent("GPU", 900, 100, 1, 1),
            MockKinetoEvent("GPU", 1000, 100, 2, 1),
            MockKinetoEvent("GPU", 1100, 100, 3, 1),
            MockKinetoEvent("GPU", 1200, 100, 4, 1),
            MockKinetoEvent("GPU", 1300, 100, 5, 1),
            MockKinetoEvent("GPU", 1700, 100, 6, 1)
        ]
        cpu_events = [
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 1, 0, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 2, 100000,
                              100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 3, 200000,
                              100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 4, 300000,
                              100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 5, 400000,
                              100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 6, 500000,
                              100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 7, 600000,
                              100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 8, 700000,
                              100000),
            MockProfilerEvent("CPU (After GPU)", 9, 800000, 100000),
            MockProfilerEvent("CPU (After GPU)", 10, 900000, 100000),
            MockProfilerEvent("CPU (After GPU)", 11, 1100000, 100000),
            MockProfilerEvent("CPU (After GPU)", 12, 1200000, 500000),
        ]

        profiler = unittest.mock.Mock()
        profiler.kineto_results = unittest.mock.Mock()
        profiler.kineto_results.events = unittest.mock.Mock(
            return_value=cuda_events)
        profiler.kineto_results.experimental_event_tree = unittest.mock.Mock(
            return_value=cpu_events)
        return profiler

    @staticmethod
    def load_mock_profile():
        accept = expecttest.ACCEPT
        json_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "profiler_utils_mock_events.json")
        if accept and torch.cuda.is_available():

            def garbage_code(x):
                for i in range(5):
                    x[0, i] = i

            x = torch.ones((4096, 4096), device="cuda")
            x = x @ x
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True) as prof:
                for _ in range(5):
                    x = x @ x
                garbage_code(x)
                for _ in range(5):
                    x = x @ x

            kineto_events = [{
                '_name':
                e.name,
                '_start_us':
                e.start_us(),
                '_duration_us':
                e.duration_us(),
                '_linked_correlation_id':
                e.linked_correlation_id(),
                '_device_type':
                1 if e.device_type() == DeviceType.CUDA else 0
            } for e in prof.profiler.kineto_results.events()]

            def EventTreeDFS(event_tree):
                from collections import deque
                stack = deque(event_tree)
                while stack:
                    curr_event = stack.pop()
                    yield curr_event
                    for child_event in curr_event.children:
                        stack.append(child_event)

            profiler_events = [{
                '_name': e.name,
                'id': e.id,
                'start_time_ns': e.start_time_ns,
                'duration_time_ns': e.duration_time_ns,
                'correlation_id': e.correlation_id,
                'children': [child.id for child in e.children],
                'parent': e.parent.id if e.parent else None
            } for e in EventTreeDFS(
                prof.profiler.kineto_results.experimental_event_tree())]

            with open(json_file_path, "w") as f:
                json.dump([kineto_events, profiler_events], f)

        assert (os.path.exists(json_file_path))
        with open(json_file_path, "r") as f:
            kineto_events, profiler_events = json.load(f)

        cuda_events = [
            MockKinetoEvent(*event.values()) for event in kineto_events
        ]
        cpu_events = []
        id_map = {}
        for e in profiler_events:
            event = MockProfilerEvent(**e)
            id_map[event.id] = event
            cpu_events.append(event)
        for event in cpu_events:
            parent = None if event.parent is None else id_map[event.parent]
            children = [id_map[child] for child in event.children]
            event.__post__init__(parent, children)
        cpu_events = [event for event in cpu_events if event.parent is None]
        profiler = unittest.mock.Mock()
        profiler.kineto_results = unittest.mock.Mock()
        profiler.kineto_results.events = unittest.mock.Mock(
            return_value=cuda_events)
        profiler.kineto_results.experimental_event_tree = unittest.mock.Mock(
            return_value=cpu_events)
        return profiler

    def test_utils_compute_self_time(self):
        with profile() as prof:
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        basic_eval = _utils.BasicEvaluation(prof.profiler)
        metrics = basic_eval.metrics
        self.assertTrue(len(metrics) > 0)
        for event_key, event_metrics in metrics.items():
            self.assertEqual(
                event_metrics.self_time_ns,
                event_key.event.duration_time_ns - sum([
                    child.duration_time_ns
                    for child in event_key.event.children
                ]))

    def test_utils_intervals_overlap(self):
        event = _utils.EventKey(MockProfilerEvent("Event 1", 1, 5, 5))
        intervals = [
            _utils.Interval(0, 9),
            _utils.Interval(1, 2),
            _utils.Interval(2, 3),
            _utils.Interval(3, 4),
            _utils.Interval(4, 5),
            _utils.Interval(8, 12),
        ]
        print(event.intervals_overlap(intervals))
        self.assertEqual(event.intervals_overlap(intervals), 5)

    def test_utils_compute_queue_depth(self):

        def format_queue_depth(queue_depth_list, events):
            res = ""
            for data, event in zip(queue_depth_list, events):
                res += f"{data.queue_depth} [{event.name}]\n"
            return res

        # We have to use Mock because time series data is too flaky to test
        profiler = self.generate_mock_profile()
        basic_evaluation = _utils.BasicEvaluation(profiler)
        self.assertExpectedInline(
            format_queue_depth(basic_evaluation.queue_depth_list,
                               basic_evaluation.cuda_events), """\
1 [cudaLaunchKernel]
2 [cudaLaunchKernel]
3 [cudaLaunchKernel]
4 [cudaLaunchKernel]
5 [cudaLaunchKernel]
4 [GPU]
3 [GPU]
2 [GPU]
1 [GPU]
0 [GPU]
1 [cudaLaunchKernel]
0 [GPU]
""")
        self.assertExpectedInline(
            format_queue_depth([
                basic_evaluation.metrics[k]
                for k in basic_evaluation.event_keys
            ], basic_evaluation.events), """\
0 [CPU (Before cudaLaunchKernel)]
0 [CPU (Before cudaLaunchKernel)]
0 [CPU (Before cudaLaunchKernel)]
0 [CPU (Before cudaLaunchKernel)]
1 [CPU (After cudaLaunchKernel)]
2 [CPU (After cudaLaunchKernel)]
3 [CPU (After cudaLaunchKernel)]
4 [CPU (After cudaLaunchKernel)]
5 [CPU (After GPU)]
4 [CPU (After GPU)]
2 [CPU (After GPU)]
1 [CPU (After GPU)]
""")

    def test_utils_compute_queue_depth_when_no_cuda_events(self):
        # For traces with only cpu events, we expect empty queue depth list
        x = torch.ones((1024, 1024))
        with profile() as prof:
            for _ in range(5):
                x = x @ x
        basic_evaluation = _utils.BasicEvaluation(prof.profiler)
        self.assertFalse(basic_evaluation.compute_queue_depth())

    def test_utils_compute_idle_time(self):
        profiler = self.generate_mock_profile()
        basic_evaluation = _utils.BasicEvaluation(profiler)
        expected_output = "\n".join([
            f"{basic_evaluation.metrics[event_key].idle_time_ns} [{event_key.event.name}]"
            for event_key in basic_evaluation.event_keys
        ])
        self.assertExpectedInline(
            expected_output, """\
100000 [CPU (Before cudaLaunchKernel)]
100000 [CPU (Before cudaLaunchKernel)]
100000 [CPU (Before cudaLaunchKernel)]
100000 [CPU (Before cudaLaunchKernel)]
0 [CPU (After cudaLaunchKernel)]
0 [CPU (After cudaLaunchKernel)]
0 [CPU (After cudaLaunchKernel)]
0 [CPU (After cudaLaunchKernel)]
0 [CPU (After GPU)]
0 [CPU (After GPU)]
0 [CPU (After GPU)]
100000 [CPU (After GPU)]""")

    @unittest.skipIf(IS_JETSON, "JSON not behaving as expected on Jetson")
    def test_utils_get_optimizable_events(self):
        basic_evaluation = _utils.BasicEvaluation(self.load_mock_profile())
        optimizable_events = basic_evaluation.get_optimizable_events(
            2, print_enable=False)
        expected_output = "\n".join(
            [f"{event_key.event.name}" for event_key in optimizable_events])
        self.assertExpectedInline(
            expected_output, """\
<built-in function _cuda_synchronize>
aten::copy_""")

    def test_profiler_name_pattern(self):
        x = torch.ones((4096, 4096))
        with profile() as prof:
            for _ in range(5):
                x = x @ x
                x = x + x
        matched_events = NamePattern(prof, "aten::mm").matched_events()
        output = "\n".join([f"{event.name}" for event in matched_events])
        self.assertExpectedInline(output, """\
aten::mm
aten::mm
aten::mm
aten::mm
aten::mm""")

    def test_profiler_pattern_match_helper(self):
        x = torch.ones((100, 100))
        with profile() as prof:
            for _ in range(5):
                x = x @ x
                x = x + x
        event_tree = prof.profiler.kineto_results.experimental_event_tree()
        pattern = Pattern(prof)
        self.assertEqual([], pattern.siblings_of(event_tree[0])[0])
        self.assertEqual(event_tree[1:], pattern.siblings_of(event_tree[0])[1])
        child_nodes = event_tree[0].children
        self.assertEqual([], pattern.siblings_of(child_nodes[0])[0])
        self.assertEqual(child_nodes[1:], pattern.siblings_of(child_nodes[0])[1])
        self.assertEqual(event_tree[0],
                         pattern.root_of(event_tree[0].children[0].children[0]))
        self.assertEqual(None, pattern.next_of(event_tree[-1]))
        self.assertEqual(event_tree[1], pattern.next_of(event_tree[0]))
        self.assertEqual(event_tree[0], pattern.prev_of(event_tree[1]))

    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite.")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_extra_cuda_copy_pattern(self):
        cases = (
            (0, lambda: torch.ones((100, 100), device="cuda")),
            (1, lambda: torch.ones((100, 100)).to("cuda")),
            (1, lambda: torch.zeros((100, 100)).to("cuda")),
            (1, lambda: torch.empty((100, 100)).fill_(5).to("cuda")),
            (1, lambda: torch.ones((100, 100)).cuda()),
            (1, lambda: torch.zeros((100, 100)).cuda()),
            (1, lambda: torch.empty((100, 100)).fill_(5).cuda()),
            (1, lambda: torch.rand((100, 100)).cuda()),
            (1, lambda: torch.randn((100, 100)).cuda()),
            (1, lambda: torch.full((100, 100), 10).cuda()),
            (0, lambda: torch.rand((100, 100)).to(dtype=torch.float16)),
            (0, lambda: torch.rand((100, 100)).half()),
            (0, lambda: torch.rand((100, 100), device="cuda").half()),
        )
        num_matched = []
        for _, fn in cases:
            with profile(with_stack=True, record_shapes=True) as prof:
                fn()
            pattern = ExtraCUDACopyPattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    @unittest.skipIf(TEST_WITH_CROSSREF,
                     "crossref intercepts calls and changes the callsite.")
    def test_profiler_for_loop_indexing_pattern(self):
        x = torch.ones((100, 100))

        def case1():
            for i in range(100):
                x[i] = i

        def case2():
            y = 0
            for i in range(100):
                y += x[i]

        def case3():
            y = 1
            for i in range(100):
                y *= x[i]

        def case4():
            y = x
            for _ in range(100):
                y = y @ x

        def case5():
            for i in range(100):
                x[i, :] = torch.arange(100) + i

        cases = ((1, case1), (1, case2), (1, case3), (0, case4), (1, case5))
        num_matched = []
        for _, fn in cases:
            with profile(with_stack=True) as prof:
                fn()
            pattern = ForLoopIndexingPattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_fp32_matmul_pattern(self):
        x = torch.ones((100, 100), device="cuda")
        with profile(with_stack=True) as prof:
            x = x @ x
        pattern = FP32MatMulPattern(prof)
        has_tf32 = 0 if pattern.skip else 1
        num_matched = len(pattern.matched_events())
        self.assertEqual(num_matched, has_tf32)


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_extra_cuda_copy_pattern_benchmark(self):
        with profile(with_stack=True, record_shapes=True) as prof:
            x = torch.ones((100, 100)).to("cuda")
            x = torch.ones((50, 50)).to("cuda")
        pattern = ExtraCUDACopyPattern(prof)
        shapes_factor_map = pattern.benchmark(pattern.matched_events())
        self.assertEqual(len(shapes_factor_map), 2)

    def test_profiler_optimizer_single_tensor_pattern(self):
        x = torch.ones((100, 100))
        cases = (
            (1, lambda: torch.optim.Adam(model.parameters())),
            (1, lambda: torch.optim.SGD(model.parameters(), lr=0.01)),
            (1, lambda: torch.optim.AdamW(model.parameters())),
            (0, lambda: torch.optim.Adam(model.parameters(), foreach=True)),
            (0, lambda: torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)),
            (0, lambda: torch.optim.AdamW(model.parameters(), foreach=True)),
        )
        num_matched = []
        for _, fn in cases:
            with profile(with_stack=True) as prof:
                model = nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 10),
                )
                optimizer = fn()
                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.cross_entropy(y_hat, torch.randint(0, 10, (100,)))
                loss.backward()
                optimizer.step()
            pattern = OptimizerSingleTensorPattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    def test_profiler_synchronized_dataloader_pattern(self):
        dataset = torch.rand((100, 100))
        sync_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        async_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=4)
        with profile(with_stack=True) as prof:
            next(iter(sync_dataloader))
            next(iter(async_dataloader))
        pattern = SynchronizedDataLoaderPattern(prof)
        num_matched = len(pattern.matched_events())
        self.assertEqual(num_matched, 1)

    def test_profiler_grad_not_set_to_none_pattern(self):
        x = torch.ones((100, 100))
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        optimizer = torch.optim.Adam(model.parameters())
        cases = (
            (0, lambda: optimizer.zero_grad()),
            (0, lambda: model.zero_grad()),
            (1, lambda: optimizer.zero_grad(set_to_none=False)),
            (1, lambda: model.zero_grad(set_to_none=False))
        )
        num_matched = []
        for _, fn in cases:
            with profile(with_stack=True) as prof:
                y_hat = model(x)
                loss = torch.nn.functional.cross_entropy(y_hat, torch.randint(0, 10, (100,)))
                loss.backward()
                optimizer.step()
                fn()
            pattern = GradNotSetToNonePattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    def test_profiler_conv2d_bias_followed_by_batchnorm2d_pattern(self):
        x = torch.randn((1, 3, 32, 32))
        cases = (
            (1, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1), nn.BatchNorm2d(3))),
            (0, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1, bias=False), nn.BatchNorm2d(3))),
            (0, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1)))
        )
        num_matched = []
        for _, model in cases:
            with profile(with_stack=True, record_shapes=True) as prof:
                model(x)
            pattern = Conv2dBiasFollowedByBatchNorm2dPattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_matmul_dim_fp16_pattern(self):
        cases = (
            (1, torch.randn((201, 201), device='cuda', dtype=torch.float16)),
            (1, torch.randn((3, 97, 97), device='cuda', dtype=torch.float16)),
            (0, torch.randn((200, 200), device='cuda', dtype=torch.float16)),
            (0, torch.randn((3, 200, 200), device='cuda', dtype=torch.float16))
        )
        num_matched = []
        for _, x in cases:
            with profile(with_stack=True, record_shapes=True) as prof:
                x @ x
            pattern = MatMulDimInFP16Pattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    def test_profiler_pattern_matcher_json_report(self):
        x = torch.ones((100, 100))
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        optimizer = torch.optim.Adam(model.parameters())
        with profile(with_stack=True, record_shapes=True) as prof:
            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, torch.randint(0, 10, (100,)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        report_all_anti_patterns(prof, json_report_dir=".", print_enable=False)
        try:
            with open("./torchtidy_report.json") as f:
                report = json.load(f)

            # It is platform dependent whether the path will include "profiler/"
            keys = [k for k in report.keys() if k.endswith("test_profiler.py")]
            self.assertEqual(len(keys), 1, f"{keys}")
            entry = report[keys[0]]

            self.assertTrue(len(entry) > 0)
            expected_fields = sorted(["line_number", "name", "url", "message"])
            for event in entry:
                actual_fields = sorted(event.keys())
                self.assertEqual(expected_fields, actual_fields)
        finally:
            os.remove("torchtidy_report.json")

if __name__ == '__main__':
    run_tests()
