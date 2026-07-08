# Owner(s): ["module: cuda graphs"]

"""Tests for CUDA graph utilities: kernel annotations and graph introspection."""

import unittest

import torch
from torch.cuda._graph_annotations import (
    _get_node_type,
    _get_stream_id,
    _is_tools_id_unavailable,
    _rekey_annotations,
    clear_kernel_annotations,
    get_kernel_annotations,
    mark_kernels,
    mark_stream,
    resolve_pending_annotations,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
)


TEST_CUDA = torch.cuda.is_available()

try:
    import cuda.bindings.runtime  # noqa: F401

    TEST_CUDA_BINDINGS = True
except ImportError:
    TEST_CUDA_BINDINGS = False


# cuda.bindings is NVIDIA-only; graph annotation APIs have no ROCm equivalent.
@instantiate_parametrized_tests
@skipIfRocm
@unittest.skipUnless(TEST_CUDA, "CUDA not available")
@unittest.skipUnless(TEST_CUDA_BINDINGS, "cuda.bindings not available")
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestMarkKernels(TestCase):
    def setUp(self):
        super().setUp()
        # Annotations are enabled per-capture via torch.cuda.graph(..,
        # enable_annotations=True); there is no global toggle.
        clear_kernel_annotations()

    def tearDown(self):
        clear_kernel_annotations()

    def test_noop_outside_capture(self):
        x = torch.randn(8, device="cuda")
        with mark_kernels("test"):
            _ = x + 1
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_memset_nodes_are_annotated(self):
        """Memset graph nodes get annotated, not just kernels and memcpys.

        A large reduction emits a ``cudaMemset`` node (zeroing the multi-block
        reduction semaphores) alongside the reduction kernel. ``keep_graph=True``
        retains the captured cudaGraph so its nodes can be classified, and defers
        the remap so node toolsIds stay keyed to the capture graph.
        """
        from cuda.bindings import driver, runtime as cuda_runtime

        memset_type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET
        big = torch.randn(8192, 8192, device="cuda")
        graph = torch.cuda.CUDAGraph(keep_graph=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("reduction"):
                _ = torch.sum(big)

        raw_graph = graph.raw_cuda_graph()
        _, num_nodes = _check_cuda_bindings(cuda_runtime.cudaGraphGetNodes(raw_graph))
        nodes, _ = _check_cuda_bindings(
            cuda_runtime.cudaGraphGetNodes(raw_graph, numNodes=num_nodes)
        )

        annotations = get_kernel_annotations()
        memset_tools_ids = [
            _check_cuda_bindings(cuda_runtime.cudaGraphNodeGetToolsId(node))
            for node in nodes[:num_nodes]
            if _get_node_type(node) == memset_type
        ]

        self.assertGreater(
            len(memset_tools_ids),
            0,
            "expected the reduction to emit at least one memset node; "
            "did the reduction lowering change?",
        )
        for tools_id in memset_tools_ids:
            self.assertIn(
                tools_id,
                annotations,
                lambda msg: f"{msg}\nmemset toolsId {hex(tools_id)} was not annotated",
            )
            self.assertEqual(annotations[tools_id], [{"str": "reduction"}])

    def test_single_scope_at_capture_start_uses_root_fallback(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            # mark_kernels is the first captured work here, so the entry
            # frontier is empty and the implementation must fall back to
            # newly created graph roots.
            with mark_kernels("phase_a"):
                _ = x + 1

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            for ann in anns:
                self.assertEqual(ann, {"str": "phase_a"})

    def test_multiple_scopes_no_overlap(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("scope_1"):
                _ = x + 1
            with mark_kernels("scope_2"):
                _ = x * 2

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
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels(annotation):
                _ = x + 1

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(anns[0]["name"], "all_gather")
            self.assertEqual(anns[0]["Group size"], 2)

    def test_clear_resets_state(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("test"):
                _ = x + 1

        self.assertGreater(len(get_kernel_annotations()), 0)
        clear_kernel_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_resolve_without_scopes_is_noop(self):
        resolve_pending_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_scope_with_no_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            _ = x + 1
            with mark_kernels("empty"):
                pass
            _ = x * 2

        for anns in get_kernel_annotations().values():
            for ann in anns:
                self.assertNotEqual(ann, "empty")

    def test_only_annotates_scope_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            _ = x + 1
            _ = x * 2
            with mark_kernels("tagged"):
                _ = x + 3
            _ = x - 1

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

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("outer"):
                _ = x + 1  # outer only
                with mark_kernels("inner"):
                    _ = x * 2  # nested: inner should win
                _ = x - 1  # outer only

        annotations = get_kernel_annotations()
        outer_ids = set()
        inner_ids = set()
        for tid, anns in annotations.items():
            self.assertEqual(
                len(anns),
                1,
                lambda msg: f"{msg}\ntoolsId {hex(tid)} has {len(anns)} annotations",
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

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels(outer_ann):
                _ = x + 1  # outer only
                with mark_kernels(inner_ann):
                    _ = x * 2  # nested
                _ = x - 1  # outer only

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

        with torch.cuda.graph(graph, enable_annotations=True):
            # Both scopes wrap the same kernels; inner exits first.
            with mark_kernels(outer_ann):
                with mark_kernels(inner_ann):
                    _ = x + 1

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

    def test_no_enable_records_nothing(self):
        # Without enable_annotations=True the capture is un-annotated, so
        # mark_kernels is a no-op and nothing is recorded.
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph):
            with mark_kernels("should_not_appear"):
                _ = x + 1

        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_enable_annotations_kwarg(self):
        """enable_annotations=True on torch.cuda.graph records and auto-resolves."""
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

        _, exec_graph_id = cuda_runtime.cudaGraphExecGetId(graph.raw_cuda_graph_exec())

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for tools_id in annotations:
            graph_id = tools_id >> 32
            self.assertEqual(
                graph_id,
                exec_graph_id,
                lambda msg: f"{msg}\ntoolsId 0x{tools_id:016x} has graph_id {graph_id}, "
                f"expected exec_graph_id {exec_graph_id}",
            )

    def test_mark_stream_snapshots_capture_before_switch(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")
        capture_stream = torch.cuda.Stream()
        aux_stream = torch.cuda.Stream()
        expected_stream_id = _get_stream_id(aux_stream)
        aux_done = torch.cuda.Event()

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=capture_stream, enable_annotations=True):
            _ = x + 1
            aux_ready = capture_stream.record_event()
            with mark_stream(aux_stream, {"name": "aux"}):
                aux_stream.wait_event(aux_ready)
                _ = x * 2
                aux_stream.record_event(aux_done)
            capture_stream.wait_event(aux_done)

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertEqual(ann["name"], "aux")
            self.assertEqual(ann["stream"], expected_stream_id)

    def test_mark_stream_annotates_target_already_capturing_when_synced(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")
        capture_stream = torch.cuda.Stream()
        aux_stream = torch.cuda.Stream()
        expected_stream_id = _get_stream_id(aux_stream)
        aux_done = torch.cuda.Event()

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=capture_stream, enable_annotations=True):
            base = x + 1
            fork_event = capture_stream.record_event()

            with torch.cuda.stream(aux_stream):
                aux_stream.wait_event(fork_event)
                _ = base * 2

            _ = base - 1
            branch_ready = capture_stream.record_event()

            with mark_stream(aux_stream, {"name": "aux_branch"}):
                aux_stream.wait_event(branch_ready)
                _ = base * 3
                aux_stream.record_event(aux_done)

            capture_stream.wait_event(aux_done)

        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for anns in annotations.values():
            self.assertEqual(len(anns), 1)
            ann = anns[0]
            self.assertEqual(ann["name"], "aux_branch")
            self.assertEqual(ann["stream"], expected_stream_id)

    def _exec_graph_id(self, graph):
        from cuda.bindings import runtime as cuda_runtime

        _, exec_graph_id = cuda_runtime.cudaGraphExecGetId(graph.raw_cuda_graph_exec())
        return exec_graph_id

    @parametrize("keep_graph", [True, False])
    def test_multiple_graphs_in_sequence_each_remapped(self, keep_graph):
        """Several graphs captured in sequence must each be annotated and
        remapped to their own exec id -- not just the last one.

        Covers both keep_graph modes: with keep_graph=True the template
        survives capture, with keep_graph=False it is destroyed -- the capture
        id is recovered from the graph object either way.
        """
        graph_a = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        graph_b = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph_a, enable_annotations=True):
            with mark_kernels("graph_a"):
                _ = x + 1

        # Break in between, then capture a second graph with its own marks.
        with torch.cuda.graph(graph_b, enable_annotations=True):
            with mark_kernels("graph_b"):
                _ = x * 2

        if keep_graph:
            # keep_graph=True defers instantiation (and thus the remap); for
            # keep_graph=False capture_end already instantiated and remapped.
            graph_a.instantiate()
            graph_b.instantiate()

        exec_a = self._exec_graph_id(graph_a)
        exec_b = self._exec_graph_id(graph_b)
        self.assertNotEqual(exec_a, exec_b)

        graph_ids_by_name: dict[str, set] = {}
        for tools_id, anns in get_kernel_annotations().items():
            self.assertEqual(len(anns), 1)
            graph_ids_by_name.setdefault(anns[0]["str"], set()).add(tools_id >> 32)

        self.assertIn("graph_a", graph_ids_by_name)
        self.assertIn("graph_b", graph_ids_by_name)
        # Each graph's annotations land under that graph's exec id, proving
        # both -- not just the last -- were remapped correctly.
        self.assertEqual(graph_ids_by_name["graph_a"], {exec_a})
        self.assertEqual(graph_ids_by_name["graph_b"], {exec_b})

    def _assert_keyed_to(self, exec_graph_id):
        annotations = get_kernel_annotations()
        self.assertGreater(len(annotations), 0)
        for tools_id in annotations:
            self.assertEqual(tools_id >> 32, exec_graph_id)

    @parametrize("trigger", ["instantiate", "replay"])
    def test_keep_graph_true_remaps_on_first_instantiation(self, trigger):
        """With keep_graph=True the exec graph is not instantiated at
        capture_end, so the context exit must not error and the remap must be
        deferred until the graph is instantiated -- whether that happens via an
        explicit instantiate() or implicitly on the first replay().
        """
        clear_kernel_annotations()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.randn(8, device="cuda")

        # Must not raise: previously __exit__ called raw_cuda_graph_exec() before
        # instantiation, which errors when keep_graph=True.
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("kg"):
                _ = x + 1

        # Annotations are recorded but still keyed by the capture id until the
        # exec graph exists.
        self.assertGreater(len(get_kernel_annotations()), 0)

        if trigger == "instantiate":
            graph.instantiate()
        else:
            graph.replay()

        self._assert_keyed_to(self._exec_graph_id(graph))

    def test_keep_graph_true_reinstantiate_rekeys_annotations(self):
        """keep_graph=True is meant for modifying the template and instantiating
        again. Each instantiate() produces a fresh exec id, so annotations must
        be rekeyed from the previous exec id to the new one on re-instantiation,
        while a plain replay() (no new exec graph) leaves them unchanged.
        """
        clear_kernel_annotations()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("kg"):
                _ = x + 1

        graph.instantiate()
        exec1 = self._exec_graph_id(graph)
        self._assert_keyed_to(exec1)

        # Replay does not create a new exec graph: keys must stay on exec1.
        graph.replay()
        self.assertEqual(self._exec_graph_id(graph), exec1)
        self._assert_keyed_to(exec1)

        # Re-instantiate: a new exec id, annotations must follow it.
        graph.instantiate()
        exec2 = self._exec_graph_id(graph)
        self.assertNotEqual(exec1, exec2)
        self._assert_keyed_to(exec2)

    def test_mark_kernels_skips_preexisting_dependents_on_entry_frontier(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")
        capture_stream = torch.cuda.Stream()
        aux_stream = torch.cuda.Stream()
        aux_done = torch.cuda.Event()

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=capture_stream, enable_annotations=True):
            base = x + 1
            shared_event = capture_stream.record_event()

            with torch.cuda.stream(aux_stream):
                aux_stream.wait_event(shared_event)
                _ = base * 2

            with mark_kernels("tagged"):
                with torch.cuda.stream(aux_stream):
                    aux_stream.wait_event(shared_event)
                    _ = base * 3
                    aux_stream.record_event(aux_done)

            capture_stream.wait_event(aux_done)

        annotations = get_kernel_annotations()
        self.assertEqual(len(annotations), 1)
        for anns in annotations.values():
            self.assertEqual(anns, [{"str": "tagged"}])


# cuda.bindings is NVIDIA-only; get_graph_data has no ROCm equivalent.
@skipIfRocm
@unittest.skipUnless(TEST_CUDA, "CUDA not available")
@unittest.skipUnless(TEST_CUDA_BINDINGS, "cuda.bindings not available")
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestGetGraphData(TestCase):
    def test_basic_structure(self):
        g = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            z = x + y
            z = z.relu()

        g.instantiate()
        data = g.get_graph_data()

        self.assertIn("exec_graph_id", data)
        self.assertIn("nodes", data)
        self.assertIsInstance(data["exec_graph_id"], int)
        self.assertGreater(len(data["nodes"]), 0)

        exec_graph_id = data["exec_graph_id"]
        for node in data["nodes"]:
            self.assertIn("index", node)
            self.assertIn("node_type", node)
            self.assertIn("tools_id", node)
            self.assertIn("graph_id", node)
            self.assertIn("node_id", node)
            self.assertIn("kernel_name", node)
            self.assertIn("dependencies", node)
            self.assertIn("dependents", node)
            self.assertEqual(node["graph_id"], exec_graph_id)
            self.assertEqual(node["tools_id"], (exec_graph_id << 32) | node["node_id"])

        kernel_nodes = [n for n in data["nodes"] if n["node_type"] == "kernel"]
        self.assertGreater(len(kernel_nodes), 0)
        for kn in kernel_nodes:
            self.assertIsNotNone(kn["kernel_name"])
            self.assertIsInstance(kn["kernel_name"], str)

    def test_export_graph_data_hook(self):
        import os
        import pickle
        import tempfile

        x = torch.zeros([2000], device="cuda")

        with tempfile.TemporaryDirectory() as tmpdir:
            # keep_graph=True: instantiate() explicitly fires the hook; the
            # template persists so we can also compare against get_graph_data().
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, capture_error_mode="relaxed"):
                _ = (x + 1).relu()
            kept_path = os.path.join(tmpdir, "kept.pkl")
            g.register_post_instantiate_hook(torch.cuda.export_graph_data(kept_path))
            g.instantiate()
            self.assertGreater(os.path.getsize(kept_path), 0)
            with open(kept_path, "rb") as f:
                self.assertEqual(pickle.load(f), g.get_graph_data())

            # keep_graph=False: capture_end instantiates (firing the hook) before
            # it destroys the template, so get_graph_data still sees both the
            # template and the exec graph. Register before capture.
            g2 = torch.cuda.CUDAGraph()
            free_path = os.path.join(tmpdir, "free.pkl")
            g2.register_post_instantiate_hook(torch.cuda.export_graph_data(free_path))
            with torch.cuda.graph(g2, capture_error_mode="relaxed"):
                _ = (x + 1).relu()
            self.assertGreater(os.path.getsize(free_path), 0)
            with open(free_path, "rb") as f:
                data = pickle.load(f)
            self.assertIn("exec_graph_id", data)
            self.assertGreater(len(data["nodes"]), 0)

    def test_edges(self):
        g = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            z = x + y
            z = z * 2
            z = z.relu()

        g.instantiate()
        data = g.get_graph_data()
        nodes = data["nodes"]

        has_deps = any(len(n["dependencies"]) > 0 for n in nodes)
        has_dependents = any(len(n["dependents"]) > 0 for n in nodes)
        self.assertTrue(has_deps)
        self.assertTrue(has_dependents)

        for node in nodes:
            for dep_idx in node["dependencies"]:
                self.assertIn(node["index"], nodes[dep_idx]["dependents"])
            for dep_idx in node["dependents"]:
                self.assertIn(node["index"], nodes[dep_idx]["dependencies"])

    def test_keep_graph_false_raises(self):
        g = torch.cuda.CUDAGraph(keep_graph=False)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            _ = x + y

        with self.assertRaises(RuntimeError):
            g.get_graph_data()


# Pure keying logic, exercised without a live capture so it runs on all of CI
# (TestMarkKernels skips unless cuda-bindings/driver >= 13.1). A toolsId packs
# the graph id in the upper 32 bits and the node id in the lower 32.
class TestRekeyAnnotations(TestCase):
    @staticmethod
    def _tools_id(graph_id, node_id):
        return (graph_id << 32) | node_id

    def test_rekeys_matching_graph(self):
        annotations = {self._tools_id(1, 10): ["a"], self._tools_id(1, 20): ["b"]}
        remapped = _rekey_annotations(annotations, capture_graph_id=1, exec_graph_id=2)
        self.assertEqual(
            remapped,
            {self._tools_id(2, 10): ["a"], self._tools_id(2, 20): ["b"]},
        )

    def test_leaves_other_graphs_untouched(self):
        other = self._tools_id(9, 10)
        annotations = {self._tools_id(1, 10): ["a"], other: ["x"]}
        remapped = _rekey_annotations(annotations, capture_graph_id=1, exec_graph_id=2)
        self.assertEqual(
            remapped,
            {self._tools_id(2, 10): ["a"], other: ["x"]},
        )

    def test_empty(self):
        self.assertEqual(
            _rekey_annotations({}, capture_graph_id=1, exec_graph_id=2), {}
        )

    def test_does_not_mutate_input_lists(self):
        original = ["a"]
        annotations = {self._tools_id(1, 10): original}
        _rekey_annotations(annotations, capture_graph_id=1, exec_graph_id=2)
        self.assertEqual(original, ["a"])


if __name__ == "__main__":
    run_tests()
