# Owner(s): ["module: cuda graphs"]

import os
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.cuda import (
    _graph_annotations,
    _graph_node_callbacks,
    _graph_py_stacks,
    graph_py_stacks,
)
from torch.cuda.graph_annotations import (
    clear_kernel_annotations,
    get_kernel_annotations,
    mark_kernels,
)
from torch.cuda.graphs import _parse_annotation_config
from torch.testing._internal.common_cuda import (
    TEST_CUDA_GRAPH_TOOLS_ID,
    TEST_CUPTI_V13_3,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


class _FakeTrace:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class TestGraphPyStacks(TestCase):
    def tearDown(self):
        graph = _graph_py_stacks._active_graph
        if graph is not None:
            _graph_py_stacks._end_capture(graph)
        super().tearDown()

    @staticmethod
    def _graph():
        return SimpleNamespace(
            _py_stack_traces={}, _py_stack_dropped=0, _remapped_exec_id=None
        )

    def test_format_stack_filters_only_pytorch_sources(self):
        internal = os.path.join(_graph_py_stacks._TORCH_ROOT, "cuda", "graphs.py")
        user = "/home/user/project/torch/model.py"
        stack = _graph_py_stacks._format_stack(
            [
                f'  File "{user}", line 12, in launch_site\n',
                f'  File "{internal}", line 34, in capture\n',
            ]
        )

        self.assertEqual(stack, "model.py:12:launch_site")
        self.assertNotIn("/home/user", stack)

    def test_format_filename_ignores_root_base(self):
        with patch.object(_graph_py_stacks, "_CWD", "/"):
            filename = _graph_py_stacks._format_filename("/workspace/project/model.py")

        self.assertEqual(filename, "model.py")

    def test_capture_is_graph_owned_and_bounded(self):
        first = self._graph()
        second = self._graph()
        traces = [_FakeTrace(), _FakeTrace(), _FakeTrace()]
        with (
            patch.object(_graph_py_stacks, "_MAX_STACKS_PER_GRAPH", 2),
            patch.object(
                _graph_py_stacks.CapturedTraceback,
                "extract",
                side_effect=traces,
            ),
        ):
            _graph_py_stacks._begin_capture(first)
            _graph_py_stacks._record(1)
            _graph_py_stacks._record(2)
            _graph_py_stacks._record(3)
            _graph_py_stacks._end_capture(first)
            _graph_py_stacks._begin_capture(second)
            _graph_py_stacks._record(4)
            _graph_py_stacks._end_capture(second)

        self.assertEqual(set(first._py_stack_traces), {1, 2})
        self.assertEqual(first._py_stack_dropped, 1)
        self.assertEqual(set(second._py_stack_traces), {4})
        first._remapped_exec_id = 1
        with (
            patch.object(
                _graph_py_stacks.CapturedTraceback,
                "format_all",
                return_value=[[], []],
            ),
            self.assertWarnsRegex(UserWarning, "1 node") as warning,
        ):
            self.assertEqual(set(graph_py_stacks.take_stacks(first)), {1, 2})
        self.assertEqual(os.path.realpath(warning.filename), os.path.realpath(__file__))

    def test_concurrent_capture_is_rejected(self):
        first = self._graph()
        second = self._graph()
        _graph_py_stacks._begin_capture(first)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            _graph_py_stacks._begin_capture(second)

    def test_remap_and_take_stacks(self):
        graph = self._graph()
        graph._capture_graph_id = 1
        graph.raw_cuda_graph_exec = lambda: 123
        trace = _FakeTrace()
        graph._py_stack_traces[(1 << 32) | 5] = trace

        runtime = SimpleNamespace(cudaGraphExecGetId=lambda _graph: (0, 0))
        with (
            patch.object(_graph_annotations, "_cuda_runtime", runtime),
            patch.object(
                _graph_annotations, "_check_cuda_bindings", side_effect=[2, 3]
            ),
            patch.object(_graph_annotations, "_kernel_annotations", {}),
            patch.object(
                _graph_py_stacks.CapturedTraceback,
                "format_all",
                return_value=[
                    ['  File "/home/user/project/model.py", line 7, in launch\n']
                ],
            ) as format_all,
        ):
            _graph_annotations.remap_to_exec_graph(graph)
            _graph_annotations.remap_to_exec_graph(graph)
            stacks = graph_py_stacks.take_stacks(graph)

        format_all.assert_called_once_with([trace])
        self.assertEqual(stacks, {(3 << 32) | 5: "model.py:7:launch"})
        self.assertEqual(graph._remapped_exec_id, 3)
        self.assertEqual(graph._py_stack_traces, {})
        self.assertTrue(trace.cleaned)

    def test_take_requires_instantiated_graph(self):
        graph = self._graph()
        graph._py_stack_traces[1] = _FakeTrace()

        with self.assertRaisesRegex(RuntimeError, "instantiate"):
            graph_py_stacks.take_stacks(graph)
        self.assertEqual(set(graph._py_stack_traces), {1})

    def test_public_module_owns_api(self):
        self.assertEqual(
            graph_py_stacks.take_stacks.__module__, "torch.cuda.graph_py_stacks"
        )

    def test_public_clear_stacks(self):
        graph = self._graph()
        trace = _FakeTrace()
        graph._py_stack_traces[1] = trace
        graph._py_stack_dropped = 2

        graph_py_stacks.clear_stacks(graph)

        self.assertEqual(graph._py_stack_traces, {})
        self.assertEqual(graph._py_stack_dropped, 0)
        self.assertTrue(trace.cleaned)

    def test_annotation_config_accepts_capture_py_stacks(self):
        config = _parse_annotation_config({"capture_py_stacks": True})
        self.assertTrue(config["capture_py_stacks"])

    def test_capture_py_stacks_requires_annotations(self):
        with self.assertRaisesRegex(ValueError, "requires enable_annotations=True"):
            torch.cuda.graph(
                self._graph(), annotation_config={"capture_py_stacks": True}
            )

    def test_callback_rejects_unexpected_domain_before_cast(self):
        handler = SimpleNamespace(domain=1, cbid=2)
        with (
            patch.object(_graph_node_callbacks, "_handler", handler),
            patch.object(_graph_py_stacks, "_record") as record,
        ):
            _graph_node_callbacks._on_graph_node_created(9, 2, object())  # type: ignore[arg-type]
        record.assert_not_called()

    def test_disarm_is_finally_safe(self):
        handler = SimpleNamespace(domain=1, cbid=2)
        monitor = SimpleNamespace(
            disarm_callback=Mock(side_effect=RuntimeError("disarm failed")),
            unregister_callback_handler=Mock(
                side_effect=RuntimeError("unregister failed")
            ),
        )
        monitor_module = SimpleNamespace(CuptiMonitor=lambda: monitor)
        warn = Mock(side_effect=RuntimeError("warning failed"))
        with (
            patch.object(_graph_node_callbacks, "_handler", handler),
            patch.object(_graph_node_callbacks, "_dropped_body_nodes", 1),
            patch.dict(sys.modules, {"torch.profiler._cupti.monitor": monitor_module}),
            patch.object(_graph_node_callbacks, "warnings", SimpleNamespace(warn=warn)),
            self.assertLogs(_graph_node_callbacks.logger, level="ERROR"),
        ):
            _graph_node_callbacks.disarm()
            self.assertIsNone(_graph_node_callbacks._handler)
            self.assertEqual(_graph_node_callbacks._dropped_body_nodes, 0)
        monitor.disarm_callback.assert_called_once_with(1, 2)
        monitor.unregister_callback_handler.assert_called_once_with(handler)
        warn.assert_called_once()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestGraphPyStacksCUDAConfig(TestCase):
    def test_auto_falls_back_when_multithreaded(self):
        x = torch.ones(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with (
            torch.autograd.grad_mode.set_multithreading_enabled(True),
            patch.object(_graph_node_callbacks, "register") as register,
        ):
            with torch.cuda.graph(graph, enable_annotations=True):
                output = x + 1

        register.assert_not_called()
        graph.replay()
        self.assertEqual(output, x + 1)

    def test_default_config_requires_single_threaded_autograd(self):
        graph = torch.cuda.CUDAGraph()
        with torch.autograd.grad_mode.set_multithreading_enabled(True):
            with self.assertRaisesRegex(
                RuntimeError, "capture_py_stacks.*single-threaded"
            ):
                with torch.cuda.graph(
                    graph,
                    enable_annotations=True,
                    annotation_config={"capture_py_stacks": True},
                ):
                    pass

    def test_registration_failure_is_actionable(self):
        graph = torch.cuda.CUDAGraph()
        with (
            torch.autograd.grad_mode.set_multithreading_enabled(False),
            patch.object(_graph_node_callbacks, "register", return_value=False),
            self.assertRaisesRegex(
                RuntimeError, "capture_py_stacks.*could not register"
            ),
        ):
            with torch.cuda.graph(
                graph,
                enable_annotations=True,
                annotation_config={"capture_py_stacks": True},
            ):
                pass
        self.assertFalse(_graph_py_stacks._is_capturing())

    def test_reservation_failure_precedes_capture(self):
        graph = torch.cuda.CUDAGraph()
        with (
            torch.autograd.grad_mode.set_multithreading_enabled(False),
            patch.object(_graph_node_callbacks, "register", return_value=True),
            patch.object(
                _graph_py_stacks, "_begin_capture", side_effect=RuntimeError("busy")
            ),
            patch.object(_graph_node_callbacks, "disarm") as disarm,
            self.assertRaisesRegex(RuntimeError, "busy"),
        ):
            with torch.cuda.graph(
                graph,
                enable_annotations=True,
                annotation_config={"capture_py_stacks": True},
            ):
                pass
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        disarm.assert_called_once_with()

    def test_post_capture_begin_failure_cleans_up(self):
        graph = torch.cuda.CUDAGraph()
        hooks = []
        graph.register_capture_end_hook(lambda _: hooks.append("capture_end"))
        graph.register_post_instantiate_hook(lambda _: hooks.append("instantiate"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
            with (
                torch.autograd.grad_mode.set_multithreading_enabled(False),
                patch.object(_graph_node_callbacks, "register", return_value=True),
                patch.object(_graph_node_callbacks, "disarm") as disarm,
                patch.object(
                    _graph_annotations,
                    "maybe_stamp_capture_root",
                    side_effect=RuntimeError("stamp failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "stamp failed"),
            ):
                with torch.cuda.graph(
                    graph,
                    enable_annotations=True,
                    annotation_config={"capture_py_stacks": True},
                ):
                    pass
        self.assertFalse(_graph_py_stacks._is_capturing())
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        self.assertEqual(hooks, [])
        with self.assertRaisesRegex(
            RuntimeError, r"capture_end\(\) must have been called"
        ):
            graph.replay()
        disarm.assert_called_once_with()

    def test_observer_disarms_between_capture_and_finalize(self):
        x = torch.ones(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        self.addCleanup(graph.reset)
        events = []
        capture_end_pre = graph.capture_end_pre
        capture_end_after_pre = graph._capture_end_after_pre

        def end_capture():
            capture_end_pre()
            events.append("capture_end")

        def finalize_capture():
            events.append("finalize")
            capture_end_after_pre()

        with (
            patch.object(graph, "capture_end_pre", side_effect=end_capture),
            patch.object(graph, "_capture_end_after_pre", side_effect=finalize_capture),
            patch.object(
                _graph_node_callbacks,
                "disarm",
                side_effect=lambda: events.append("disarm"),
            ),
        ):
            with torch.cuda.graph(
                graph,
                enable_annotations=True,
                annotation_config={"backend": "edge_walk"},
            ):
                output = x + 1

        self.assertEqual(events[:3], ["capture_end", "disarm", "finalize"])
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output, x + 1)

    def test_finalize_failure_restores_stream(self):
        original_stream = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        self.addCleanup(graph.reset)

        with (
            patch.object(
                graph,
                "_capture_end_after_pre",
                side_effect=RuntimeError("finalize failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "finalize failed"),
        ):
            with torch.cuda.graph(graph, stream=capture_stream):
                torch.ones(1, device="cuda") + 1

        self.assertEqual(torch.cuda.current_stream(), original_stream)

    def test_arm_failure_warns_and_cleans_up(self):
        x = torch.ones(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with (
            torch.autograd.grad_mode.set_multithreading_enabled(False),
            patch.object(_graph_node_callbacks, "register", return_value=True),
            patch.object(_graph_node_callbacks, "arm", return_value=False),
            patch.object(_graph_node_callbacks, "disarm") as disarm,
            self.assertWarnsRegex(UserWarning, "could not arm"),
        ):
            with torch.cuda.graph(
                graph,
                enable_annotations=True,
                annotation_config={"capture_py_stacks": True},
            ):
                output = x + 1

        graph.replay()
        self.assertEqual(output, x + 1)
        self.assertFalse(_graph_py_stacks._is_capturing())
        self.assertEqual(graph_py_stacks.take_stacks(graph), {})
        self.assertGreaterEqual(disarm.call_count, 1)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
@unittest.skipUnless(
    TEST_CUDA_GRAPH_TOOLS_ID,
    "requires cudaGraphNodeGetToolsId (cuda-bindings and driver >= 13.1)",
)
class TestGraphPyStacksCUDA(TestCase):
    def setUp(self):
        super().setUp()
        clear_kernel_annotations()
        self.addCleanup(clear_kernel_annotations)
        ctx = torch.autograd.grad_mode.set_multithreading_enabled(False)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    @staticmethod
    def _warm(x):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                x = x.sin()
        torch.cuda.current_stream().wait_stream(stream)
        return x

    def test_stacks_match_annotations_and_graph_nodes(self):
        def launch_site(x):
            return x.sin()

        x = self._warm(torch.randn(64, device="cuda"))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(
            graph,
            enable_annotations=True,
            annotation_config={"backend": "cupti", "capture_py_stacks": True},
        ):
            with mark_kernels("phase"):
                output = launch_site(x)

        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()

        self.assertIsNone(_graph_node_callbacks._handler)
        stacks = graph_py_stacks.take_stacks(graph)
        self.assertEqual(set(stacks), set(get_kernel_annotations()))
        self.assertGreater(len(stacks), 0)
        graph_data = graph.get_graph_data()
        node_tools_ids = {node["tools_id"] for node in graph_data["nodes"]}
        self.assertTrue(set(stacks).issubset(node_tools_ids))
        self.assertEqual(
            {tools_id >> 32 for tools_id in stacks}, {graph_data["exec_graph_id"]}
        )
        self.assertTrue(any(":launch_site" in stack for stack in stacks.values()))
        self.assertEqual(output, x.sin())

    def test_default_instantiation_remaps_stacks(self):
        x = self._warm(torch.randn(64, device="cuda"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            enable_annotations=True,
            annotation_config={"capture_py_stacks": True},
        ):
            output = x.cos()

        self.assertIsNone(_graph_node_callbacks._handler)
        stacks = graph_py_stacks.take_stacks(graph)
        self.assertGreater(len(stacks), 0)
        self.assertEqual(
            {tools_id >> 32 for tools_id in stacks}, {graph._remapped_exec_id}
        )
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output, x.cos())

    def test_edge_walk_annotations_with_stacks(self):
        x = self._warm(torch.randn(64, device="cuda"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            enable_annotations=True,
            annotation_config={"backend": "edge_walk", "capture_py_stacks": True},
        ):
            with mark_kernels("phase"):
                output = x.exp()

        stacks = graph_py_stacks.take_stacks(graph)
        self.assertEqual(set(stacks), set(get_kernel_annotations()))
        self.assertGreater(len(stacks), 0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output, x.exp())

    def test_stack_capture_counts_conditional_body_nodes(self):
        from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

        x = torch.ones(2048, device="cuda")
        pred = torch.tensor(True, device="cuda")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with self.assertWarnsRegex(UserWarning, "node observation skipped") as warning:
            with torch.cuda.graph(
                graph,
                enable_annotations=True,
                annotation_config={"backend": "edge_walk", "capture_py_stacks": True},
            ):
                y = x + 1
                with _if_body(pred):
                    _ = y.sqrt()

        self.assertEqual(os.path.realpath(warning.filename), os.path.realpath(__file__))
        graph.instantiate()
        stacks = graph_py_stacks.take_stacks(graph)
        self.assertGreater(len(stacks), 0)
        exec_graph_id = graph.get_graph_data()["exec_graph_id"]
        self.assertEqual({tools_id >> 32 for tools_id in stacks}, {exec_graph_id})


if __name__ == "__main__":
    run_tests()
