# Owner(s): ["module: inductor"]

import contextlib
import copy
import json
import os
import tempfile
import unittest

import torch
from torch._inductor import config
from torch._inductor.profiler import _InductorTraceProcessor, inductor_trace_handler
from torch.profiler import kineto_available, profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.inductor_utils import GPU_TYPE


class InductorProfilerTraceTests(TestCase):
    @contextlib.contextmanager
    def _kernel_information_jsons(self, kernel_information_jsons):
        import torch._inductor.debug as inductor_debug

        previous = copy.deepcopy(inductor_debug.get_kernel_information_jsons())
        current = inductor_debug.get_kernel_information_jsons()
        current.clear()
        current.update(kernel_information_jsons)
        try:
            yield
        finally:
            current.clear()
            current.update(previous)

    def test_build_flow_mapping_supports_sparse_flow_ids(self):
        flow_id = 1_000_000_000
        trace = {
            "traceEvents": [
                {
                    "name": "aten::add",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 5,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "ac2g",
                    "cat": "ac2g",
                    "ph": "s",
                    "id": flow_id,
                    "ts": 5,
                    "tid": 1,
                    "args": {},
                },
                {
                    "name": "triton_poi_fused_add_0",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 10,
                    "dur": 7,
                    "tid": 2,
                    "args": {},
                },
                {
                    "name": "ac2g",
                    "cat": "ac2g",
                    "ph": "f",
                    "id": flow_id,
                    "ts": 17,
                    "tid": 2,
                    "args": {},
                },
            ]
        }

        prof = _InductorTraceProcessor()
        prof._assign_uniq_id_to_event(trace)

        src2dst, dst2src = prof._build_flow_mapping(
            trace,
            [trace["traceEvents"][1], trace["traceEvents"][3]],
        )

        self.assertEqual(src2dst, {0: 2})
        self.assertEqual(dst2src, {2: 0})

    def test_maybe_triton_call_uses_inductor_prefix(self):
        prof = _InductorTraceProcessor()

        self.assertTrue(prof._maybe_triton_call("triton_poi_fused_add_0"))
        self.assertFalse(prof._maybe_triton_call("not_triton_related"))

    def test_add_inductor_kernel_stack_to_chrome_trace(self):
        kernel_name = "triton_poi_fused_add_0"
        stack = ["model.py:7 in forward"]
        kernel_information_jsons = {
            str(("Torch-Compiled Region: 0/0", False)): {
                kernel_name: {
                    "stack_traces": stack,
                    "post_grad_nodes": ["add"],
                    "pre_grad_nodes": ["add"],
                }
            }
        }
        with self._kernel_information_jsons(kernel_information_jsons):
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
                        "id": 0,
                        "ts": 15,
                        "tid": 1,
                        "args": {},
                    },
                    {
                        "name": kernel_name,
                        "cat": "cuda_driver",
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
                        "id": 0,
                        "ts": 27,
                        "tid": 2,
                        "args": {},
                    },
                ]
            }

            prof = _InductorTraceProcessor()
            updated_trace = prof.add_to_chrome_trace(trace)

            self.assertEqual(updated_trace["traceEvents"][3]["args"]["stack"], stack)

    def test_add_inductor_kernel_stack_to_chrome_trace_by_external_id(self):
        kernel_name = "triton_poi_fused_add_0"
        stack = ["model.py:7 in forward"]
        kernel_information_jsons = {
            str(("Torch-Compiled Region: 0/0", False)): {
                kernel_name + ":1": {
                    "stack_traces": stack,
                    "post_grad_nodes": ["add"],
                    "pre_grad_nodes": ["add"],
                }
            }
        }
        with self._kernel_information_jsons(kernel_information_jsons):
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
                        "name": kernel_name,
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 10,
                        "dur": 5,
                        "tid": 1,
                        "args": {"External id": 6},
                    },
                    {
                        "name": kernel_name,
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 20,
                        "dur": 3,
                        "tid": 2,
                        "args": {"External id": 6},
                    },
                ]
            }

            prof = _InductorTraceProcessor()
            updated_trace = prof.add_to_chrome_trace(trace)

            self.assertEqual(updated_trace["traceEvents"][2]["args"]["stack"], stack)

    def test_add_inductor_kernel_stack_to_chrome_trace_by_external_id_without_triton_kernel_name(
        self,
    ):
        kernel_name = "triton_poi_fused_add_0"
        stack = ["model.py:7 in forward"]
        kernel_information_jsons = {
            str(("Torch-Compiled Region: 0/0", False)): {
                kernel_name + ":1": {
                    "stack_traces": stack,
                    "post_grad_nodes": ["add"],
                    "pre_grad_nodes": ["add"],
                }
            }
        }
        with self._kernel_information_jsons(kernel_information_jsons):
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
                        "name": "hipLaunchKernel",
                        "cat": "cuda_runtime",
                        "ph": "X",
                        "ts": 10,
                        "dur": 5,
                        "tid": 1,
                        "args": {"External id": 6},
                    },
                    {
                        "name": "kernel",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 20,
                        "dur": 3,
                        "tid": 2,
                        "args": {"External id": 6},
                    },
                ]
            }

            prof = _InductorTraceProcessor()
            updated_trace = prof.add_to_chrome_trace(trace)

            self.assertEqual(updated_trace["traceEvents"][2]["args"]["stack"], stack)

    def test_add_inductor_kernel_stack_to_chrome_trace_skips_missing_stack(self):
        kernel_name = "triton_poi_fused_add_0"
        kernel_information_jsons = {
            str(("Torch-Compiled Region: 0/0", False)): {
                kernel_name: {
                    "post_grad_nodes": ["add"],
                    "pre_grad_nodes": ["add"],
                }
            }
        }
        with self._kernel_information_jsons(kernel_information_jsons):
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
                        "id": 0,
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
                        "id": 0,
                        "ts": 27,
                        "tid": 2,
                        "args": {},
                    },
                ]
            }

            prof = _InductorTraceProcessor()
            updated_trace = prof.add_to_chrome_trace(trace)

            self.assertNotIn("stack", updated_trace["traceEvents"][3]["args"])

    def test_add_inductor_kernel_stack_to_chrome_trace_backward_quoted_region(self):
        compile_name = "Torch-Compiled Region: odd', True) name"
        kernel_name = "triton_poi_fused_mul_0"
        stack = ["model.py:9 in backward"]
        kernel_information_jsons = {
            str((compile_name + "_backward_0", True)): {
                kernel_name: {
                    "stack_traces": stack,
                    "post_grad_nodes": ["mul"],
                    "pre_grad_nodes": ["mul"],
                }
            }
        }
        with self._kernel_information_jsons(kernel_information_jsons):
            trace = {
                "traceEvents": [
                    {
                        "name": compile_name,
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
                        "name": "fwdbwd",
                        "cat": "fwdbwd",
                        "ph": "s",
                        "id": 0,
                        "ts": 16,
                        "tid": 1,
                        "args": {},
                    },
                    {
                        "name": "CompiledFunctionBackward",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 200,
                        "dur": 100,
                        "tid": 1,
                        "args": {},
                    },
                    {
                        "name": "fwdbwd",
                        "cat": "fwdbwd",
                        "ph": "f",
                        "id": 0,
                        "ts": 201,
                        "tid": 1,
                        "args": {},
                    },
                    {
                        "name": "aten::mul",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 210,
                        "dur": 5,
                        "tid": 1,
                        "args": {"External id": 6},
                    },
                    {
                        "name": kernel_name,
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 220,
                        "dur": 3,
                        "tid": 2,
                        "args": {"External id": 6},
                    },
                ]
            }

            prof = _InductorTraceProcessor()
            updated_trace = prof.add_to_chrome_trace(trace)

            self.assertEqual(updated_trace["traceEvents"][6]["args"]["stack"], stack)

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_compile_timeline_provenance_survives_reset_scope(self):
        import torch._inductor.debug as inductor_debug

        previous_kernel_information_jsons = copy.deepcopy(
            inductor_debug.get_kernel_information_jsons()
        )
        inductor_debug.get_kernel_information_jsons().clear()

        def fn(x):
            return torch.sin(x + 1).relu()

        try:
            with (
                config.patch(
                    {
                        "force_disable_caches": True,
                        "trace.provenance_tracking_to_timeline": True,
                        "triton.unique_kernel_names": True,
                    }
                ),
                tempfile.TemporaryDirectory() as trace_dir,
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(64, 64, device=GPU_TYPE)
                activity = getattr(ProfilerActivity, GPU_TYPE.upper())
                with profile(
                    activities=[ProfilerActivity.CPU, activity],
                    on_trace_ready=inductor_trace_handler(
                        trace_dir, worker_name="worker"
                    ),
                ):
                    compiled_fn(x)
                    torch.get_device_module(GPU_TYPE).synchronize()

                trace_files = [
                    filename
                    for filename in os.listdir(trace_dir)
                    if filename.endswith(".pt.trace.json")
                ]
                self.assertEqual(len(trace_files), 1)
                with open(os.path.join(trace_dir, trace_files[0])) as f:
                    trace = json.load(f)

            kernel_stacks = [
                event.get("args", {}).get("stack")
                for event in trace["traceEvents"]
                if event.get("cat") == "kernel"
            ]
            self.assertTrue(
                any(stack for stack in kernel_stacks),
                "Expected a generated kernel in the exported trace to carry a stack",
            )
            self.assertEqual(inductor_debug.get_kernel_information_jsons(), {})
        finally:
            inductor_debug.get_kernel_information_jsons().clear()
            inductor_debug.get_kernel_information_jsons().update(
                previous_kernel_information_jsons
            )


if __name__ == "__main__":
    run_tests()
