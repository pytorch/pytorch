# Owner(s): ["module: inductor"]

import gc
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch._inductor.profiler as inductor_profiler
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.profiler import (
    _add_inductor_kernel_stacks,
    _build_flow_mapping,
    _export_inductor_trace,
    _find_events_covered_in,
    _register_compiled_graph,
    _registered_graph_provenance,
    inductor_trace_handler,
)
from torch._inductor.utils import fresh_cache
from torch.profiler import kineto_available, profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


class _FakeCompiledGraph:
    def __init__(self, kernel_info):
        self.inductor_provenance_stack_traces_str = json.dumps(kernel_info)


class _FakeProfiler:
    def __init__(self, trace):
        self.trace = trace

    def export_chrome_trace(self, path, use_python_export=False):
        with open(path, "w") as f:
            json.dump(self.trace, f)


class InductorProfilerTraceTests(TestCase):
    def _trace_with_flow(self, kernel_name="triton_poi_fused_add_0"):
        return {
            "traceEvents": [
                {
                    "name": "Torch-Compiled Region: 0/0",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "## Call CompiledFxGraph graph_key ##",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "aten::add",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 10,
                    "dur": 5,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "ac2g",
                    "cat": "ac2g",
                    "ph": "s",
                    "id": 7,
                    "ts": 15,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": kernel_name,
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 20,
                    "dur": 7,
                    "tid": 2,
                    "args": {},
                },
                {
                    "name": "ac2g",
                    "cat": "ac2g",
                    "ph": "f",
                    "id": 7,
                    "ts": 27,
                    "tid": 2,
                    "args": {},
                },
            ]
        }

    def _graph_provenance(self, kernel_name="triton_poi_fused_add_0"):
        return {"graph_key": {kernel_name + ":1": ["model.py:7 in forward"]}}

    def test_flow_ids_are_scoped_by_type(self):
        flow_id = 1_000_000_000
        events = [
            {"name": "forward", "cat": "cpu_op"},
            {"name": "fwdbwd", "cat": "fwdbwd", "ph": "s", "id": flow_id},
            {"name": "backward", "cat": "cpu_op"},
            {"name": "fwdbwd", "cat": "fwdbwd", "ph": "f", "id": flow_id},
            {"name": "launch", "cat": "cuda_runtime"},
            {"name": "ac2g", "cat": "ac2g", "ph": "s", "id": flow_id},
            {"name": "kernel", "cat": "kernel"},
            {"name": "ac2g", "cat": "ac2g", "ph": "f", "id": flow_id},
        ]
        indexed_events = list(enumerate(events))
        fwdbwd = _build_flow_mapping(indexed_events, "fwdbwd")
        ac2g = _build_flow_mapping(indexed_events, "ac2g")

        self.assertEqual(fwdbwd, ({0: 2}, {2: 0}))
        self.assertEqual(ac2g, ({4: 6}, {6: 4}))

    def test_equal_timestamp_uses_innermost_region(self):
        outer = {"ts": 0, "dur": 100, "tid": 1}
        inner = {"ts": 0, "dur": 50, "tid": 1}
        event = {"ts": 0, "dur": 50, "tid": 1}

        covered = _find_events_covered_in([(2, event)], [(0, outer), (1, inner)])

        self.assertEqual(covered, {1: {2}})

    def test_add_inductor_kernel_stack(self):
        trace = self._trace_with_flow()

        updated_trace = _add_inductor_kernel_stacks(trace, self._graph_provenance())

        self.assertEqual(
            updated_trace["traceEvents"][4]["args"]["stack"],
            ["model.py:7 in forward"],
        )
        self.assertNotIn("uid_assigned", updated_trace)
        self.assertNotIn("uid", updated_trace["traceEvents"][0])

    def test_add_extern_kernel_stack_by_external_id(self):
        wrapper_name = "extern_kernels_torch_ops_aten_mm_default_1"
        trace = {
            "traceEvents": [
                {
                    "name": "Torch-Compiled Region: 0/0",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "## Call CompiledFxGraph graph_key ##",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": wrapper_name,
                    "cat": "python_function",
                    "ph": "X",
                    "ts": 10,
                    "dur": 20,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "aten::mm",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 12,
                    "dur": 5,
                    "tid": 1,
                    "args": {"External id": 6},
                },
                {
                    "name": "ampere_sgemm_32x32",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 40,
                    "dur": 3,
                    "tid": 2,
                    "args": {"External id": 6},
                },
            ]
        }
        provenance = {"graph_key": {wrapper_name: ["model.py:9 in forward"]}}

        _add_inductor_kernel_stacks(trace, provenance)

        self.assertEqual(
            trace["traceEvents"][4]["args"]["stack"],
            ["model.py:9 in forward"],
        )

    def test_registered_provenance_is_reusable_and_weak(self):
        graph = _FakeCompiledGraph(self._graph_provenance()["graph_key"])
        _register_compiled_graph("test_reusable_graph", graph)

        first = _registered_graph_provenance()["test_reusable_graph"]
        second = _registered_graph_provenance()["test_reusable_graph"]
        self.assertEqual(first, second)

        del graph
        gc.collect()
        self.assertNotIn("test_reusable_graph", _registered_graph_provenance())

    def test_export_adds_registered_stack(self):
        graph = _FakeCompiledGraph(self._graph_provenance()["graph_key"])
        _register_compiled_graph("graph_key", graph)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.json")
            with config.patch(
                {
                    "cpp_wrapper": False,
                    "trace.provenance_tracking_max_trace_size": 0,
                    "triton.unique_kernel_names": True,
                }
            ):
                _export_inductor_trace(_FakeProfiler(self._trace_with_flow()), path)

            with open(path) as f:
                trace = json.load(f)
            self.assertEqual(
                trace["traceEvents"][4]["args"]["stack"],
                ["model.py:7 in forward"],
            )

    def test_export_preserves_raw_trace_on_processing_failure(self):
        raw_trace = self._trace_with_flow()
        graph = _FakeCompiledGraph(self._graph_provenance()["graph_key"])
        _register_compiled_graph("graph_key", graph)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.json")
            with (
                config.patch(
                    {
                        "cpp_wrapper": False,
                        "trace.provenance_tracking_max_trace_size": 0,
                    }
                ),
                mock.patch.object(
                    inductor_profiler,
                    "_add_inductor_kernel_stacks",
                    side_effect=RuntimeError("failed"),
                ),
                self.assertLogs("torch._inductor.profiler", level="ERROR"),
            ):
                _export_inductor_trace(_FakeProfiler(raw_trace), path)

            with open(path) as f:
                self.assertEqual(json.load(f), raw_trace)

    def test_trace_size_guard_runs_before_json_load(self):
        raw_trace = self._trace_with_flow()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.json")
            with (
                config.patch("trace.provenance_tracking_max_trace_size", 1),
                mock.patch.object(inductor_profiler.json, "load") as load,
                self.assertLogs("torch._inductor.profiler", level="WARNING"),
            ):
                _export_inductor_trace(_FakeProfiler(raw_trace), path)

            load.assert_not_called()
            with open(path) as f:
                self.assertEqual(json.load(f), raw_trace)

    @requires_gpu_and_triton
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "force_disable_caches": False,
            "trace.provenance_tracking_to_timeline": True,
            "triton.unique_kernel_names": True,
        }
    )
    @functorch_config.patch("enable_autograd_cache", False)
    def test_cache_hit_has_provenance_in_repeated_traces(self):
        self.addCleanup(torch._dynamo.reset)

        def fn(x):
            return torch.sin(x + 1).relu()

        device_module = torch.get_device_module(GPU_TYPE)
        x = torch.randn(64, 64, device=GPU_TYPE)
        with fresh_cache(), tempfile.TemporaryDirectory() as trace_dir:
            torch._dynamo.reset()
            counters.clear()
            torch.compile(fn, backend="inductor")(x)
            device_module.synchronize()

            torch._dynamo.reset()
            cache_hits = counters["inductor"]["fxgraph_cache_hit"]
            compiled_fn = torch.compile(fn, backend="inductor")
            handler = inductor_trace_handler(trace_dir, worker_name="worker")
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())
            with profile(
                activities=[ProfilerActivity.CPU, activity],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=2
                ),
                on_trace_ready=handler,
            ) as prof:
                for _ in range(2):
                    compiled_fn(x)
                    device_module.synchronize()
                    prof.step()

            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], cache_hits)
            trace_files = sorted(os.listdir(trace_dir))
            self.assertEqual(len(trace_files), 2)
            for trace_file in trace_files:
                with open(os.path.join(trace_dir, trace_file)) as f:
                    trace = json.load(f)
                stacks = [
                    event.get("args", {}).get("stack")
                    for event in trace["traceEvents"]
                    if event.get("cat") == "kernel"
                ]
                self.assertTrue(any(stacks))


if __name__ == "__main__":
    run_tests()
