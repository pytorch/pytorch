# Owner(s): ["module: cuda graphs"]

"""Tests for CUDA graph kernel annotation via mark_kernels."""

import unittest

import torch
from torch.cuda._graph_annotations import (
    _is_tools_id_unavailable,
    clear_kernel_annotations,
    enable_annotations,
    get_kernel_annotations,
    mark_kernels,
    remap_to_exec_graph,
    resolve_pending_annotations,
)
from torch.testing._internal.common_utils import run_tests, TestCase


TEST_CUDA = torch.cuda.is_available()

try:
    import cuda.bindings.runtime  # noqa: F401

    TEST_CUDA_BINDINGS = True
except ImportError:
    TEST_CUDA_BINDINGS = False


@unittest.skipUnless(TEST_CUDA, "CUDA not available")
@unittest.skipUnless(TEST_CUDA_BINDINGS, "cuda.bindings not available")
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestMarkKernels(TestCase):
    def setUp(self):
        enable_annotations()
        clear_kernel_annotations()

    def tearDown(self):
        clear_kernel_annotations()

    def test_noop_outside_capture(self):
        x = torch.randn(8, device="cuda")
        with mark_kernels("test"):
            _ = x + 1
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_single_scope(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("phase_a"):
                _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, {"str": "phase_a"})

    def test_multiple_scopes_no_overlap(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("scope_1"):
                _ = x + 1
            with mark_kernels("scope_2"):
                _ = x * 2
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        scope_1_ids = set()
        scope_2_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(len(anns), 1)
            if anns[0] == {"str": "scope_1"}:
                scope_1_ids.add(tid)
            elif anns[0] == {"str": "scope_2"}:
                scope_2_ids.add(tid)

        self.assertGreater(len(scope_1_ids), 0)
        self.assertGreater(len(scope_2_ids), 0)
        self.assertEqual(len(scope_1_ids & scope_2_ids), 0)

    def test_dict_annotation(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        annotation = {"name": "all_gather", "Group size": 2, "dtype": "bfloat16"}
        with torch.cuda.graph(graph):
            with mark_kernels(annotation):
                _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(anns[0]["name"], "all_gather")
            self.assertEqual(anns[0]["Group size"], 2)

    def test_clear_resets_state(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("test"):
                _ = x + 1
            resolve_pending_annotations()

        self.assertGreater(len(get_kernel_annotations()), 0)
        clear_kernel_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_resolve_without_scopes_is_noop(self):
        resolve_pending_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_scope_with_no_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            _ = x + 1
            with mark_kernels("empty"):
                pass
            _ = x * 2
            resolve_pending_annotations()

        for anns in get_kernel_annotations().values():
            for ann in anns:
                self.assertNotEqual(ann, "empty")

    def test_only_annotates_scope_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            _ = x + 1
            _ = x * 2
            with mark_kernels("tagged"):
                _ = x + 3
            _ = x - 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        total_annotated = sum(len(anns) for anns in annotations.values())
        self.assertGreater(total_annotated, 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, {"str": "tagged"})

    def test_nested_scopes_innermost_wins(self):
        """With nested string scopes, the innermost name wins."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("outer"):
                _ = x + 1  # outer only
                with mark_kernels("inner"):
                    _ = x * 2  # nested: inner should win
                _ = x - 1  # outer only
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        outer_ids = set()
        inner_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(
                len(anns), 1, f"toolsId {hex(tid)} has {len(anns)} annotations"
            )
            ann = anns[0]
            self.assertIsInstance(ann, dict)
            if ann["str"] == "outer":
                outer_ids.add(tid)
            elif ann["str"] == "inner":
                inner_ids.add(tid)

        self.assertGreater(len(outer_ids), 0, "Should have outer-only kernels")
        self.assertGreater(len(inner_ids), 0, "Should have inner kernels")
        self.assertEqual(len(outer_ids & inner_ids), 0)

    def test_nested_dict_scopes_inner_wins_common_keys(self):
        """With truly nested dict scopes, inner wins for common keys,
        outer-only keys are preserved."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        outer_ann = {"name": "ag_collective", "stream": 71}
        inner_ann = {
            "name": "all_gather",
            "stream": 62,
            "In msg nelems": 1024,
            "dtype": "bfloat16",
        }

        with torch.cuda.graph(graph):
            with mark_kernels(outer_ann):
                _ = x + 1  # outer only
                with mark_kernels(inner_ann):
                    _ = x * 2  # nested
                _ = x - 1  # outer only
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        outer_only_ids = set()
        nested_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertIsInstance(ann, dict)
            if ann["name"] == "ag_collective":
                outer_only_ids.add(tid)
            elif ann["name"] == "all_gather":
                nested_ids.add(tid)
                # Inner wins for common keys
                self.assertEqual(ann["stream"], 62)
                # Inner-only keys preserved
                self.assertEqual(ann["In msg nelems"], 1024)
                self.assertEqual(ann["dtype"], "bfloat16")

        self.assertGreater(len(outer_only_ids), 0, "Should have outer-only kernels")
        self.assertGreater(len(nested_ids), 0, "Should have nested kernels")

    def test_same_range_scopes_inner_wins_common_keys(self):
        """With same-range scopes (inner ctx exits first), inner wins
        for common keys, outer-only keys are preserved."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        outer_ann = {"name": "ag_collective", "stream": 71}
        inner_ann = {
            "name": "all_gather",
            "stream": 62,
            "In msg nelems": 1024,
            "dtype": "bfloat16",
        }

        with torch.cuda.graph(graph):
            # Both scopes wrap the same kernels; inner exits first.
            with mark_kernels(outer_ann):
                with mark_kernels(inner_ann):
                    _ = x + 1
            resolve_pending_annotations()

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertIsInstance(ann, dict)
            # Inner wins for common keys
            self.assertEqual(ann["name"], "all_gather", "Inner name should win")
            self.assertEqual(ann["stream"], 62, "Inner stream should win")
            # Inner-only keys preserved
            self.assertEqual(ann["In msg nelems"], 1024)
            self.assertEqual(ann["dtype"], "bfloat16")

    def test_remap_to_exec_graph(self):
        from cuda.bindings import runtime as cuda_runtime

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("test"):
                _ = x + 1
            resolve_pending_annotations()

        annotations_before = dict(get_kernel_annotations())
        self.assertGreater(len(annotations_before), 0)

        exec_handle = cuda_runtime.cudaGraphExec_t(
            init_value=graph.raw_cuda_graph_exec()
        )
        _, exec_graph_id = cuda_runtime.cudaGraphExecGetId(exec_handle)

        remap_to_exec_graph(graph)

        annotations_after = get_kernel_annotations()
        self.assertEqual(len(annotations_after), len(annotations_before))
        for tools_id in annotations_after:
            self.assertEqual(tools_id >> 32, exec_graph_id)

    def test_disabled_is_noop(self):
        from torch.cuda._graph_annotations import disable_annotations

        disable_annotations()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("should_not_appear"):
                _ = x + 1
            resolve_pending_annotations()

        self.assertEqual(len(get_kernel_annotations()), 0)

        # Re-enable for other tests
        enable_annotations()

    def test_enable_annotations_kwarg(self):
        """enable_annotations on torch.cuda.graph auto-resolves annotations."""
        from torch.cuda._graph_annotations import disable_annotations

        # Start with annotations disabled to verify the kwarg enables them.
        disable_annotations()
        clear_kernel_annotations()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("auto"):
                _ = x + 1

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, {"str": "auto"})

    def test_enable_annotations_does_not_clear(self):
        """Annotations from a previous graph survive a second capture."""
        from torch.cuda._graph_annotations import disable_annotations

        disable_annotations()
        clear_kernel_annotations()

        graph1 = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph1, enable_annotations=True):
            with mark_kernels("first"):
                _ = x + 1

        first_count = len(get_kernel_annotations())
        self.assertGreater(first_count, 0)

        graph2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph2, enable_annotations=True):
            with mark_kernels("second"):
                _ = x * 2

        # Both graphs' annotations should be present.
        self.assertGreater(len(get_kernel_annotations()), first_count)

    def test_enable_annotations_remaps_to_exec_graph(self):
        """enable_annotations=True must remap toolsIds to the exec graph ID."""
        from cuda.bindings import runtime as cuda_runtime

        clear_kernel_annotations()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("remap_test"):
                _ = x + 1

        exec_handle = cuda_runtime.cudaGraphExec_t(
            init_value=graph.raw_cuda_graph_exec()
        )
        _, exec_graph_id = cuda_runtime.cudaGraphExecGetId(exec_handle)

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for tools_id in annotations:
            graph_id = tools_id >> 32
            self.assertEqual(
                graph_id,
                exec_graph_id,
                f"toolsId 0x{tools_id:016x} has graph_id {graph_id}, "
                f"expected exec_graph_id {exec_graph_id}",
            )

    def test_enable_annotations_false_does_not_auto_resolve(self):
        """Without enable_annotations, pending scopes are not resolved."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        # enable_annotations=False (default): no auto-resolve.
        with torch.cuda.graph(graph):
            with mark_kernels("unresolved"):
                _ = x + 1

        # Annotations should be empty because resolve was never called.
        self.assertEqual(len(get_kernel_annotations()), 0)


if __name__ == "__main__":
    run_tests()
