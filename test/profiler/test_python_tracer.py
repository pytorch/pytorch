# Owner(s): ["oncall: profiler"]

import collections
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import unittest

import torch
import torch.nn as nn
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch.autograd.profiler import profile as _profile
from torch.profiler import (
    _utils,
    kineto_available,
    profile,
    ProfilerActivity,
    record_function,
)
from torch.testing._internal.common_profiler import (
    _profiler_test_skip_reason,
    initialize_kineto_with_accelerator,
    skipIfProfilerUnsupported,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    serialTest,
    skipIfPythonVersionMismatch,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_WITH_CROSSREF,
    TestCase,
)


def setUpModule():
    if _profiler_test_skip_reason() is None:
        initialize_kineto_with_accelerator()


# The tracer sees every live Python thread, including tqdm's monitor.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass


@instantiate_parametrized_tests
class TestPythonTracer(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_method_with_c_function(self):
        class A:
            method_with_c_function = classmethod(repr)

        def get_key(x):
            A().method_with_c_function()
            time.sleep(1.2)
            return len(x)

        names = ["Alice", "Bob"]

        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            sorted(names, key=get_key)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
                found = False
                for event in events:
                    if (
                        event.get("cat", "") == "python_function"
                        and event.get("name", "") == "<built-in function sorted>"
                    ):
                        duration = event.get("dur", 0)
                        if duration >= 2000000:
                            found = True
                            break
                self.assertTrue(found)

    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_monitoring_callback(self):
        vi = sys.version_info
        from sys import monitoring

        with profile(activities=[ProfilerActivity.CPU], with_stack=True):
            name = monitoring.get_tool(2)
            if vi.micro < 5:
                self.assertEqual(name, "PyTorch Profiler")
            else:
                self.assertEqual(name, None)
        name = monitoring.get_tool(2)
        self.assertEqual(name, None)

    def test_unexpected_c_return_events(self):
        code = """
import threading
import time
import torch

from threading import Event, Lock

lock = Lock()
lock.acquire()
event1 = Event()
event2 = Event()
event3 = Event()

def run():
    event1.set()
    event2.wait()
    lock.acquire()
    event3.set()

threading.Thread(target=run).start()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    event1.wait()
    event2.set()
    time.sleep(1)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    lock.release()
    event3.wait()
    """

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        self.assertFalse(
            "Python replay stack is empty during pop operation" in result.stderr
        )

    @skipIfProfilerUnsupported
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    # Avoid automatic inlining so scripted calls remain in the recorded stack.
    @torch.jit.optimized_execution(False)
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode"""

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

        source_file = os.path.basename(__file__)
        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any(source_file in entry for entry in e.stack))
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

        if kineto_available():
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

    @skipIfProfilerUnsupported
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
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
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
                    if idx in threads:
                        raise AssertionError(f"Thread index {idx} already exists")
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
            len(tid_counts),
            expected_threads,
            lambda msg: f"{msg}\n{expected_threads}, {tid_counts}",
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

    @skipIfProfilerUnsupported
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_python_gc_event(self):
        activities = [ProfilerActivity.CPU]

        def payload():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            with record_function("pre_gc"):
                torch.mm(x, y)
            gc.collect()
            with record_function("post_gc"):
                torch.mm(x, y)

        def validate_json(prof, gc_collection_on):
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    # Find required events
                    if gc_collection_on:
                        pre_gc = next(
                            (e for e in events if e["name"] == "pre_gc"), None
                        )
                        post_gc = next(
                            (e for e in events if e["name"] == "post_gc"), None
                        )
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        # Assert all required events are present
                        self.assertIsNotNone(pre_gc, "pre_gc event is missing")
                        self.assertIsNotNone(post_gc, "post_gc event is missing")
                        self.assertTrue(
                            len(python_gc_events) > 0, "No Python GC events found"
                        )
                        # Calculate boundaries
                        pre_gc_end = pre_gc["ts"] + pre_gc.get("dur", 0)
                        post_gc_start = post_gc["ts"]
                        # Assert at least one Python GC event is correctly placed.
                        # Other automatic GC events can happen while the profiler is
                        # active, especially during first-run initialization.
                        python_gc_events_between = [
                            e
                            for e in python_gc_events
                            if e["ts"] > pre_gc_end
                            and e["ts"] + e.get("dur", 0) < post_gc_start
                        ]
                        self.assertTrue(
                            len(python_gc_events_between) > 0,
                            "No Python GC events found between pre_gc and post_gc",
                        )
                    else:
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        self.assertTrue(
                            len(python_gc_events) == 0,
                            "Python GC event found when flag off",
                        )

        for gc_flag in [True, False]:
            with profile(
                activities=activities,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    record_python_gc_info=gc_flag
                ),
                with_stack=True,
            ) as prof:
                payload()
            validate_json(prof, gc_flag)


if __name__ == "__main__":
    run_tests()
