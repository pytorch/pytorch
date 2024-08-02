# Owner(s): ["oncall: profiler"]

import collections
import gc
import json
import mmap
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import expecttest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
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
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_JETSON,
    IS_LINUX,
    IS_WINDOWS,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_WITH_ASAN,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
)


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

try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):
    @skipCUDAVersionIn([(11, 5)])  # https://github.com/pytorch/pytorch/issues/69023
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA"""
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
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))
        )
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(
            not (is_increasing and max_diff > 100 * 1024),
            msg=f"memory usage is increasing, {str(last_rss)}",
        )

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
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
        p = subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
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
""",
            ],
            universal_newlines=True,
            timeout=60,
        )

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
                (x,) = ctx.saved_tensors
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


@instantiate_parametrized_tests
class TestProfiler(TestCase):
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode"""
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
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            return mod(x)

        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
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
                self.assertTrue(
                    any(
                        (
                            "test_source" in entry
                            or "ts_method_1" in entry
                            or "ts_method_2" in entry
                        )
                        for entry in e.stack
                    )
                )

        # TODO: https://github.com/pytorch/kineto/issues/617
        if kineto_available() and not IS_WINDOWS:
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    self.assertEqual(
                        len(matches), 1, repr([e["name"] for e in matches])
                    )
                    return matches[0]

                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                self.assertEqual(
                    module_event["args"]["Python parent id"],
                    wrapper_event["args"]["Python id"],
                )

        torch._C._set_graph_executor_optimize(prev_opt)

    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            "multiple_preexisting": ((False, False),) * 2,
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
        name_fn=lambda name, thread_spec: name,
    )
    @serialTest()
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
            def __init__(self) -> None:
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
        nodes = [
            node
            for node in _utils.traverse_dfs(roots)
            if isinstance(node.extra_fields, _ExtraFields_PyCall)
        ]
        tid_counts = collections.Counter([node.start_tid for node in nodes])

        prior_threads = sum(
            not start_under_profiler for start_under_profiler, _ in thread_spec
        )
        expected_threads = prior_threads + 1
        self.assertEqual(
            len(tid_counts), expected_threads, f"{expected_threads}, {tid_counts}"
        )
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2**64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [
            node.start_tid
            for node in nodes
            if "known_preexisting_function" in node.name
        ]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [
            node.start_tid for node in nodes if "invoked_during_run" in node.name
        ]
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

    def _check_stats(self, profiler_stats):
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        # rerun to avoid initial start overhead
        with _profile(use_cuda=use_cuda, use_kineto=True) as p:
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name.lower() or "Cijk" in e.name:
                found_gemm = True
            if "memcpy" in e.name.lower():
                found_memcpy = True
        if use_cuda:
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        self._check_stats(p._stats)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_kineto_multigpu(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
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
        self._check_stats(prof._stats())

    def test_memory_profiler(self):
        def run_profiler(tensor_creation_fn):
            # collecting allocs / deallocs
            with _profile(
                profile_memory=True,
                record_shapes=True,
                use_kineto=kineto_available(),
            ) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            # print(stats)
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            # print(stat_metrics)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertGreater(
                        stat_metrics[alloc_fn], 0, f"alloc_fn = {alloc_fn}"
                    )
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertLess(
                        stat_metrics[dealloc_fn], 0, f"alloc_fn = {dealloc_fn}"
                    )

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
            ],
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
                with open(fname) as f:
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
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.backends.mkldnn.is_available():
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
                ],
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
            allocs=["aten::rand", "aten::empty"],
            deallocs=["[memory]"],
        )
        if torch.cuda.is_available():
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])

    @unittest.skipIf(
        IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared"
    )
    def test_oom_tracing(self):
        def run_profiler(tensor_creation_fn):
            with _profile(profile_memory=True, record_shapes=True) as prof:
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        def create_cuda_tensor_oom():
            device = torch.device("cuda:0")
            return torch.empty(
                1024, 1024, 1024, 1024, dtype=torch.float32, device=device
            )

        def check_trace(fname):
            prof.export_chrome_trace(fname)
            with open(fname) as f:
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
            def __init__(self) -> None:
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
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."
        ]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.",
            "TOP(C)::forward.",
        ]
        with TemporaryFileName(mode="w+") as fname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_modules=True,
            ) as prof:
                model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with open(fname) as f:
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
        """Checks that python side high level events are recorded."""

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
        criterion = torch.nn.MSELoss(reduction="sum")
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
                        actual_event_count[key] = (
                            actual_event_count.setdefault(key, 0) + 1
                        )
            for key, count in expected_event_count.items():
                self.assertTrue(
                    (key in actual_event_count.keys())
                    and (count == actual_event_count[key])
                )

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N,
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
            "Optimizer.zero_grad#CustomSGD.zero_grad": N,
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
        nested_tensor = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 5))], layout=torch.jagged
        )
        with _profile(
            record_shapes=True, with_flops=True, use_kineto=kineto_available()
        ) as prof:
            model(inputs)
            # test that nested tensor won't cause exception during flop compute
            nested_tensor = nested_tensor + nested_tensor
        profiler_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        self.assertIn("Total MFLOPs", profiler_output)
        if not (kineto_available() and torch.cuda.is_available()):
            return

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1
        )
        self.assertIn("Total MFLOPs", profiler_output)

    def test_kineto_profiler_api(self):
        called_num = [0]

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
                row_limit=-1,
            )
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        initial_step = KinetoStepTracker.current_step()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            for idx in range(8):
                self.payload(use_cuda=use_cuda)
                p.step()

        self.assertEqual(called_num[0], 2)
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        # case without schedule
        with profile(activities=supported_activities()) as p:
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)

        test_schedule = torch.profiler.schedule(
            skip_first=2, wait=1, warmup=1, active=2, repeat=2
        )
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
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        ) as p:
            for idx in range(niters):
                run_batch()
                p.step()

        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 2 * niters)

    def test_export_stacks(self):
        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with open(fname) as f:
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
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-3:], ["pt", "trace", "json"])
                file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            p = profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dname, use_gzip=True
                ),
            )
            p.start()
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            p.stop()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-5].isdigit() and int(parts[-5]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-4:], ["pt", "trace", "json", "gz"])
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
            with open(fname) as f:
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
            with open(fname) as f:
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
            with open(fname) as f:
                json.load(f)

    def test_profiler_tracing(self):
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    def test_profiler_op_event_args(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        with _profile(record_shapes=True) as prof:
            a = torch.ones((64, 32), dtype=torch.float32)
            c = torch.cat([a, a]).sin()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::ones":
                        self.assertEqual(
                            args["Input type"],
                            ["ScalarList", "Scalar", "", "", "Scalar"],
                        )
                        self.assertEqual(
                            args["Concrete Inputs"], ["[64, 32]", "6", "", "", "False"]
                        )

                    if e["name"] == "aten::cat":
                        self.assertEqual(args["Input Dims"], [[[64, 32], [64, 32]], []])
                        self.assertEqual(args["Input type"], ["TensorList", "Scalar"])

                    # check that each op has record function id
                    self.assertGreaterEqual(
                        args.get("Record function id", -1),
                        0,
                        f"Failed finding record funciont for op = {e}",
                    )

    def test_profiler_strides(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        with _profile(record_shapes=True) as prof:
            c = torch.add(a, b)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::add":
                        self.assertEqual(args["Input Strides"], [[17, 1], [25, 2], []])

    def test_profiler_fwd_bwd_link(self):
        with _profile(use_kineto=True) as prof:
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True
            )
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                events = j["traceEvents"]
                ts_to_name = {}
                flow_s_to_ts = {}
                flow_f_to_ts = {}
                for e in events:
                    if e["ph"] == "X":
                        ts_to_name[e["ts"]] = e["name"]
                    if (
                        "cat" in e
                        and "name" in e
                        and e["cat"] == "fwdbwd"
                        and e["name"] == "fwdbwd"
                    ):
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
                self.assertTrue(
                    all(
                        ts in ts_to_name.keys()
                        for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2]
                    )
                )
                self.assertTrue(
                    ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits"
                )
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")

    def test_profiler_disable_fwd_bwd_link(self):
        try:
            torch._C._profiler._set_fwd_bwd_enabled_val(False)

            with _profile(use_kineto=True) as prof:
                t1, t2 = torch.ones(1, requires_grad=True), torch.ones(
                    1, requires_grad=True
                )
                z = torch.add(t1, t2)
                y = torch.ones(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                loss.backward()

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    events = j["traceEvents"]

                    for e in events:
                        self.assertNotEqual(e.get("cat", None), "fwdbwd")
        finally:
            torch._C._profiler._set_fwd_bwd_enabled_val(True)

    # This test is broken on Windows, the likely reason is that kineto/CUPTI
    # is not supported that particular environment. Once the CI stabilizes
    # we can narrow the condition so Windows is checked as well (TODO)
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(IS_WINDOWS, "Test does not work on Windows")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_cuda_sync_events(self):
        device = torch.device("cuda:0")
        t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)

        def workload() -> None:
            torch.add(t1, t2)
            torch.cuda.synchronize()
            torch.add(t1, t2)

        def trace_and_check(exp_config: Optional[_ExperimentalConfig]) -> None:
            with _profile(
                use_kineto=True,
                use_cuda=True,
                experimental_config=exp_config,
            ) as prof:
                workload()

            with TemporaryFileName(mode="w+") as fname:
                # fname = "/tmp/kineto_out.json"
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    cats = {e.get("cat", None) for e in j["traceEvents"]}
            self.assertTrue(
                "cuda_sync" in cats,
                "Expected to find cuda_sync event" f" found = {cats}",
            )

        print("Testing enable_cuda_sync_events in _ExperimentalConfig")
        trace_and_check(exp_config=_ExperimentalConfig(enable_cuda_sync_events=True))

        print("Testing _profiler._set_cuda_sync_enabled_val()")
        try:
            torch._C._profiler._set_cuda_sync_enabled_val(True)
            trace_and_check(exp_config=None)
        finally:
            torch._C._profiler._set_cuda_sync_enabled_val(False)

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
        """
        We expect the correlation_id to be unique across multiple invokation of the profiler,
        So we will reuse id_uniqueness_set.
        """
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
                if (corr_id) and event.device_type() == DeviceType.CPU:
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
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_with_environment_variable(self):
        script = """
import torch
import torch.nn as nn
from torch.profiler import supported_activities, profile
from torch.autograd.profiler import KinetoStepTracker

class SimpleNet(nn.Module):
    def __init__(self) -> None:
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
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )

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
            for before, after in [(True, False), (False, True)]:
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

    def test_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            for _ in range(4):
                # Test first with no optional args
                with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf1"]), 4
        )
        for e in p.events():
            if e.name == "add_test_fast_rf1":
                self.assertTrue(e.input_shapes == [])
                self.assertTrue(e.kwinputs == {})
        with profile(record_shapes=True) as p:
            # add optional args
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf2", [x, y], {"stream": 0, "grid": "lambda x : x + 1"}
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf2"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf2":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])
                self.assertTrue(e.kwinputs == {"stream": 0, "grid": "lambda x : x + 1"})

        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf3", input_values=["hi"], keyword_values={"hi": "hello"}
            )
            for _ in range(4):
                try:
                    with cm:
                        x.add(y)
                        raise ValueError
                        x.relu()
                except ValueError:
                    pass

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf3"]), 4
        )
        self.assertFalse(any((e.name and "relu" in e.name) for e in p.events()))

        for e in p.events():
            if e.name == "add_test_fast_rf3":
                self.assertTrue(e.input_shapes == [[]])

        with profile() as p:
            for _ in range(4):
                with torch._C._profiler._RecordFunctionFast(
                    "add_test_fast_rf4", [x, y]
                ):
                    x.add(y)
                    with torch._C._profiler._RecordFunctionFast("add_test_fast_rf5"):
                        x.relu()

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf4"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf4":
                self.assertTrue(e.input_shapes == [])

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf5"]), 4
        )

        with profile(record_shapes=True) as p:
            # test optional args with tuple
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf6",
                (
                    x,
                    y,
                ),
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf6"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf6":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {"stream": 0, "grid": "lambda x : x + 1", "debug": 'debug"'},
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    if e["name"] == "add_test_kwinputs":
                        args = e["args"]
                        self.assertTrue("stream" in args)
                        self.assertTrue("grid" in args)
                        self.assertTrue(args["stream"] == "0")
                        self.assertTrue(args["grid"] == "lambda x : x + 1")
                        self.assertTrue(args["debug"] == "None")

    def test_is_profiler_enabled(self):
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with torch.autograd.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

    def test_guarded_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))

        with profile() as p:
            cm = torch._C._profiler._RecordFunctionFast("guarded_rff")
            for _ in range(4):
                if torch.autograd.profiler._is_profiler_enabled:
                    with cm:
                        x.add(y)
                else:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "guarded_rff"]), 4
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_event_list(self):
        # AFAIK event list is part of legacy profiler and/or used when kineto is not available.
        # This test has basic sanity checks to test against obvious regressions.
        x, y = (torch.rand((4, 4), requires_grad=True, device="cuda") for _ in range(2))
        with profile(with_stack=True) as p:
            z = (x @ y).relu().sum()
            z.backward()

        event_list = torch.autograd.profiler_util.EventList(p.events())
        # event_list._build_tree()

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                json.load(f)

        event_list.table()

    def _check_all_gpu_present(self, gpu_dict, max_gpu_count):
        for i in range(0, max_gpu_count):
            self.assertEqual(gpu_dict["GPU " + str(i)], 1)

    # Do json sanity testing. Checks that all events are between profiler start and end
    # also checks to see that GPU values are present in trace if cuda is used
    def _validate_basic_json(self, traceEvents, cuda_available=False):
        MAX_GPU_COUNT = 8
        PROFILER_IDX = -4
        RECORD_END = -1
        RECORD_START = -2
        traceEventProfiler = traceEvents[PROFILER_IDX]

        self.assertTrue(traceEventProfiler["name"] == "PyTorch Profiler (0)")
        self.assertTrue(traceEvents[RECORD_END]["name"] == "Record Window End")
        self.assertTrue(
            traceEvents[RECORD_START]["name"] == "Iteration Start: PyTorch Profiler"
        )
        # check that the profiler starts/ends within the record interval
        self.assertGreaterEqual(
            traceEventProfiler["ts"],
            traceEvents[RECORD_START]["ts"],
            "Profiler starts before record!",
        )
        self.assertLessEqual(
            traceEventProfiler["ts"] + traceEventProfiler["dur"],
            traceEvents[RECORD_END]["ts"],
            "Profiler ends after record end!",
        )

        gpu_dict = collections.defaultdict(int)
        for i, traceEvent in enumerate(traceEvents):
            if (
                i == len(traceEvents) + RECORD_END
                or i == len(traceEvents) + RECORD_START
            ):
                continue
            # make sure all valid trace events are within the bounds of the profiler
            if "ts" in traceEvent:
                self.assertGreaterEqual(
                    traceEvent["ts"],
                    traceEventProfiler["ts"],
                    "Trace event is out of bounds",
                )
            # some python events seem to go a little past record end probably because
            # of some clock inaccuracies so just compare events ending to RECORD_END
            if "dur" in traceEvent:
                self.assertLessEqual(
                    traceEvent["ts"] + traceEvent["dur"],
                    traceEvents[RECORD_END]["ts"],
                    "Trace event ends too late!",
                )
            gpu_value = traceEvent.get("args", {}).get("labels", None)
            if gpu_value and "GPU" in gpu_value:
                gpu_dict[gpu_value] += 1
                # Max PID offset is 5M, based from pytorch/kineto include header:
                # https://github.com/pytorch/kineto/blob/8681ff11e1fa54da39023076c5c43eddd87b7a8a/libkineto/include/output_base.h#L35
                kExceedMaxPid = 5000000
                self.assertTrue(
                    traceEvents[i + 1]["args"]["sort_index"]
                    == kExceedMaxPid + int(gpu_value.split()[1])
                )

        # TODO add checking gpu count if cpuOnly_ is true or not

    def _test_chrome_trace_basic_helper(self, with_cuda=False):
        if with_cuda:
            device = "cuda"
        else:
            device = "cpu"
        x, y = (torch.rand(4, 4).to(device) for _ in range(2))

        with profile(with_stack=True) as p:
            torch.add(x, y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
                self._validate_basic_json(report["traceEvents"], with_cuda)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_chrome_trace(self):
        self._test_chrome_trace_basic_helper()
        if torch.cuda.is_available():
            self._test_chrome_trace_basic_helper(with_cuda=True)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_time_scale(self):
        MARGIN_ERROR = 0.5
        SEC_TO_US = 1000 * 1000
        WAIT_TIME = 10
        with profile() as p:
            with torch.profiler.record_function("test_span"):
                for i in range(WAIT_TIME):
                    torch.rand(4, 4)
                    time.sleep(1)
        events = p.events()

        # make sure function events are scaled appropriately
        self.assertTrue(events[0].name == "test_span")
        test_span = events[0]
        self.assertGreaterEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME - MARGIN_ERROR,
            "event out of range",
        )
        self.assertLessEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME + MARGIN_ERROR,
            "event out of range",
        )

        # make sure tracing is scaled appropriately
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
            events = report["traceEvents"]
            for event in events:
                if event["name"] == "test_span":
                    self.assertGreaterEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME - MARGIN_ERROR,
                        "profiling out of range",
                    )
                    self.assertLessEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME + MARGIN_ERROR,
                        "profiling out of range",
                    )

    def _schedule_helper(self, warmup, active, repeat):
        with profile(
            schedule=torch.profiler.schedule(
                skip_first=0, wait=0, warmup=warmup, active=active, repeat=repeat
            )
        ) as prof:
            for i in range(100):
                torch.add(1, 2)
                prof.step()
        for ev in prof.key_averages():
            if ev.key == "aten::add":
                return ev.count
        return 0

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_schedule_function_count(self):
        self.assertEqual(self._schedule_helper(warmup=0, active=1, repeat=1), 1)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=0), 100)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=10), 50)
        self.assertEqual(self._schedule_helper(warmup=1, active=5, repeat=0), 83)
        self.assertEqual(self._schedule_helper(warmup=10, active=10, repeat=4), 40)
        self.assertEqual(self._schedule_helper(warmup=50, active=1, repeat=0), 1)

    def _step_helper_func(self, prof):
        time.sleep(0.1)
        torch.randn(1, 3, 224, 224)
        prof.step()

    def _partial_overlap(self, prof_step, step_helper_func):
        p_start = prof_step["ts"]
        p_end = prof_step["ts"] + prof_step["dur"]
        h_start = step_helper_func["ts"]
        h_end = step_helper_func["ts"] + step_helper_func["dur"]

        if p_start < h_start and p_end < h_end and p_end > h_start:
            return True
        if p_start > h_start and p_start < h_end and p_end > h_end:
            return True
        return False

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_cpu_annotation_overlap(self):
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
        ) as prof:
            for i in range(5):
                self._step_helper_func(prof)
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            prof_steps = []
            step_helper_funcs = []
            with open(fname) as f:
                report = json.load(f)
                for event in report["traceEvents"]:
                    if "ProfilerStep" in event["name"]:
                        prof_steps.append(event)
                    if "step_helper_func" in event["name"]:
                        step_helper_funcs.append(event)
            self.assertEqual(len(prof_steps), 5)
            self.assertEqual(len(step_helper_funcs), 5)
            for i in range(0, len(step_helper_funcs)):
                for j in range(0, len(step_helper_funcs)):
                    self.assertTrue(
                        not self._partial_overlap(prof_steps[i], step_helper_funcs[j])
                    )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_user_annotation(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()) as p:
            with torch.profiler.record_function("test_user_annotation"):
                self.payload(use_cuda=use_cuda)

        for evt in p.key_averages():
            if evt.key == "test_user_annotation":
                self.assertTrue(evt.is_user_annotation)
            else:
                self.assertFalse(evt.is_user_annotation)


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


@dataclass(frozen=True)
class MockKinetoEvent:
    _name: str
    _start_us: int
    _duration_us: int
    _linked_correlation_id: int
    _device_type: int

    @property
    def name(self) -> str:
        return self._name

    def start_ns(self) -> int:
        return self._start_us * 1000

    def duration_ns(self) -> int:
        return self._duration_us * 1000

    def linked_correlation_id(self) -> int:
        return self._linked_correlation_id

    def device_type(self) -> DeviceType:
        return DeviceType.CUDA if self._device_type == 1 else DeviceType.CPU


@dataclass(frozen=True)
class MockProfilerEvent:
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
                "1": {"2": {}},
                "3": {
                    "4": {},
                    "5": {},
                },
            },
            "root_1": {
                "6": {},
                "7": {},
                "8": {
                    "9": {"10": {}},
                },
            },
        }
        return [MockNode(name, i) for name, i in tree.items()]

    def test_dfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_dfs(self.make_tree())),
            "root_0 1 2 3 4 5 root_1 6 7 8 9 10",
        )

    def test_bfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_bfs(self.make_tree())),
            "root_0 root_1 1 3 6 7 8 2 4 5 9 10",
        )

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
            MockKinetoEvent("GPU", 1700, 100, 6, 1),
        ]
        cpu_events = [
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 1, 0, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 2, 100000, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 3, 200000, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 4, 300000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 5, 400000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 6, 500000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 7, 600000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 8, 700000, 100000),
            MockProfilerEvent("CPU (After GPU)", 9, 800000, 100000),
            MockProfilerEvent("CPU (After GPU)", 10, 900000, 100000),
            MockProfilerEvent("CPU (After GPU)", 11, 1100000, 100000),
            MockProfilerEvent("CPU (After GPU)", 12, 1200000, 500000),
        ]

        profiler = unittest.mock.Mock()
        profiler.kineto_results = unittest.mock.Mock()
        profiler.kineto_results.events = unittest.mock.Mock(return_value=cuda_events)
        profiler.kineto_results.experimental_event_tree = unittest.mock.Mock(
            return_value=cpu_events
        )
        return profiler

    @staticmethod
    def load_mock_profile():
        accept = expecttest.ACCEPT
        json_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "profiler_utils_mock_events.json",
        )
        if accept and torch.cuda.is_available():

            def garbage_code(x):
                for i in range(5):
                    x[0, i] = i

            x = torch.ones((4096, 4096), device="cuda")
            x = x @ x
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                for _ in range(5):
                    x = x @ x
                garbage_code(x)
                for _ in range(5):
                    x = x @ x

            kineto_events = [
                {
                    "_name": e.name,
                    "_start_ns": e.start_ns(),
                    "_duration_ns": e.duration_ns(),
                    "_linked_correlation_id": e.linked_correlation_id(),
                    "_device_type": 1 if e.device_type() == DeviceType.CUDA else 0,
                }
                for e in prof.profiler.kineto_results.events()
            ]

            def EventTreeDFS(event_tree):
                from collections import deque

                stack = deque(event_tree)
                while stack:
                    curr_event = stack.pop()
                    yield curr_event
                    for child_event in curr_event.children:
                        stack.append(child_event)

            profiler_events = [
                {
                    "_name": e.name,
                    "id": e.id,
                    "start_time_ns": e.start_time_ns,
                    "duration_time_ns": e.duration_time_ns,
                    "correlation_id": e.correlation_id,
                    "children": [child.id for child in e.children],
                    "parent": e.parent.id if e.parent else None,
                }
                for e in EventTreeDFS(
                    prof.profiler.kineto_results.experimental_event_tree()
                )
            ]

            with open(json_file_path, "w") as f:
                json.dump([kineto_events, profiler_events], f)

        assert os.path.exists(json_file_path)
        with open(json_file_path) as f:
            kineto_events, profiler_events = json.load(f)

        cuda_events = [MockKinetoEvent(*event.values()) for event in kineto_events]
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
        profiler.kineto_results.events = unittest.mock.Mock(return_value=cuda_events)
        profiler.kineto_results.experimental_event_tree = unittest.mock.Mock(
            return_value=cpu_events
        )
        return profiler

    def test_utils_compute_self_time(self):
        with profile() as prof:
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True
            )
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
                event_key.event.duration_time_ns
                - sum(child.duration_time_ns for child in event_key.event.children),
            )

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
            format_queue_depth(
                basic_evaluation.queue_depth_list, basic_evaluation.cuda_events
            ),
            """\
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
""",
        )
        self.assertExpectedInline(
            format_queue_depth(
                [basic_evaluation.metrics[k] for k in basic_evaluation.event_keys],
                basic_evaluation.events,
            ),
            """\
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
""",
        )

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
        expected_output = "\n".join(
            [
                f"{basic_evaluation.metrics[event_key].idle_time_ns} [{event_key.event.name}]"
                for event_key in basic_evaluation.event_keys
            ]
        )
        self.assertExpectedInline(
            expected_output,
            """\
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
100000 [CPU (After GPU)]""",
        )

    @unittest.skipIf(IS_JETSON, "JSON not behaving as expected on Jetson")
    def test_utils_get_optimizable_events(self):
        basic_evaluation = _utils.BasicEvaluation(self.load_mock_profile())
        optimizable_events = basic_evaluation.get_optimizable_events(
            2, print_enable=False
        )
        expected_output = "\n".join(
            [f"{event_key.event.name}" for event_key in optimizable_events]
        )
        self.assertExpectedInline(
            expected_output,
            """\
<built-in function _cuda_synchronize>
aten::copy_""",
        )

    def test_profiler_name_pattern(self):
        x = torch.ones((4096, 4096))
        with profile() as prof:
            for _ in range(5):
                x = x @ x
                x = x + x
        matched_events = NamePattern(prof, "aten::mm").matched_events()
        output = "\n".join([f"{event.name}" for event in matched_events])
        self.assertExpectedInline(
            output,
            """\
aten::mm
aten::mm
aten::mm
aten::mm
aten::mm""",
        )

    # TODO: Add logic for CUDA version of test
    @unittest.skipIf(torch.cuda.is_available(), "Test not working for CUDA")
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
        self.assertEqual(
            event_tree[0], pattern.root_of(event_tree[0].children[0].children[0])
        )
        self.assertEqual(None, pattern.next_of(event_tree[-1]))
        self.assertEqual(event_tree[1], pattern.next_of(event_tree[0]))
        self.assertEqual(event_tree[0], pattern.prev_of(event_tree[1]))

    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
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

    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
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
                loss = torch.nn.functional.cross_entropy(
                    y_hat, torch.randint(0, 10, (100,))
                )
                loss.backward()
                optimizer.step()
            pattern = OptimizerSingleTensorPattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    def test_profiler_synchronized_dataloader_pattern(self):
        dataset = torch.rand((100, 100))
        sync_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        async_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=10, num_workers=4
        )
        with profile(with_stack=True) as prof:
            next(iter(sync_dataloader))
            next(iter(async_dataloader))
        pattern = SynchronizedDataLoaderPattern(prof)
        num_matched = len(pattern.matched_events())
        self.assertEqual(num_matched, 1)

    @skipIfTorchDynamo(
        "pattern checks for aten::_zero op which might not be there with torch.compile'd graph"
    )
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
            (1, lambda: model.zero_grad(set_to_none=False)),
        )
        num_matched = []
        for _, fn in cases:
            with profile(with_stack=True) as prof:
                y_hat = model(x)
                loss = torch.nn.functional.cross_entropy(
                    y_hat, torch.randint(0, 10, (100,))
                )
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
            (0, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1))),
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
            (1, torch.randn((201, 201), device="cuda", dtype=torch.float16)),
            (1, torch.randn((3, 97, 97), device="cuda", dtype=torch.float16)),
            (0, torch.randn((200, 200), device="cuda", dtype=torch.float16)),
            (0, torch.randn((3, 200, 200), device="cuda", dtype=torch.float16)),
        )
        num_matched = []
        for _, x in cases:
            with profile(with_stack=True, record_shapes=True) as prof:
                x @ x
            pattern = MatMulDimInFP16Pattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
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
            loss = torch.nn.functional.cross_entropy(
                y_hat, torch.randint(0, 10, (100,))
            )
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

    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "x86 linux only cpp unwinding")
    def test_fuzz_symbolize(self):
        # generate some random addresses in the text section and make sure the
        # symbolizers do not throw exceptions/crash
        def get_text_sections():
            text_sections = []
            seen = set()
            for filename in os.listdir("/proc/self/map_files"):
                library = os.readlink("/proc/self/map_files/" + filename)
                if ".so" not in library or library in seen:
                    continue
                seen.add(library)
                with open(os.path.join("/proc/self/map_files", library), "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

                    def unpack(fmt, offset):
                        return struct.unpack(
                            fmt, mm[offset : offset + struct.calcsize(fmt)]
                        )

                    if mm[:4] != b"\x7fELF":
                        continue
                    (section_headers_start,) = unpack("Q", 40)
                    (section_header_size,) = unpack("H", 58)
                    (num_section_headers,) = unpack("H", 60)
                    (shstrndx,) = unpack("H", 62)
                    (shstrtab_offset,) = unpack(
                        "Q", section_headers_start + shstrndx * section_header_size + 24
                    )
                    for i in range(num_section_headers):
                        (section_name_offset,) = unpack(
                            "I", section_headers_start + i * section_header_size
                        )
                        name_start = shstrtab_offset + section_name_offset
                        section_name = mm[name_start : name_start + 6]
                        if section_name != b".text\0":
                            continue
                        (section_offset,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 24
                        )
                        (section_size,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 32
                        )
                        start = int(filename.split("-")[0], 16) + section_offset
                        text_sections.append((start, section_size))
                        break
                    mm.close()
            return text_sections

        r = random.Random()
        r.seed(1)
        text_sections = get_text_sections()
        addrs = []
        for i in range(200):
            s = r.randrange(0, len(text_sections))
            start, size = text_sections[s]
            addr = r.randrange(start, start + size)
            addrs.append(addr)
        fast = torch._C._profiler.symbolize_addresses(addrs, "fast")
        dladdr = torch._C._profiler.symbolize_addresses(addrs, "dladdr")
        addr2line = torch._C._profiler.symbolize_addresses(addrs, "addr2line")
        self.assertEqual(len(fast), len(addrs))
        self.assertEqual(len(addr2line), len(fast))


if __name__ == "__main__":
    run_tests()
