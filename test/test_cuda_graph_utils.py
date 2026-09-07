# Owner(s): ["module: cuda graphs"]

"""Tests for CUDA graph utilities: kernel annotations and graph introspection."""

import unittest
import unittest.mock

import torch
from torch.cuda._graph_annotations import (
    _get_annotatable_types,
    _get_node_type,
    _get_stream_id,
    _is_tools_id_unavailable,
    _rekey_annotations,
    _reset_kernel_annotations,
    mark_stream,
    resolve_and_remap,
    resolve_pending_annotations,
)
from torch.cuda._utils import _check_cuda_bindings, _check_cuda_bindings_driver
from torch.cuda.graph_annotations import (
    clear_kernel_annotations,
    get_kernel_annotations,
    is_available,
    mark_kernels,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TestCase,
)


# Nested graphs (child-graph and conditional nodes) have no torch API, so the tests
# below build them through cuda.bindings the way a library embedding one would.
# cuda.bindings is imported lazily so this module still imports without it.
def _memset_params(dst: torch.Tensor):
    from cuda.bindings import runtime as cuda_runtime

    params = cuda_runtime.cudaMemsetParams()
    params.dst = dst.data_ptr()
    params.elementSize = 4
    params.width = dst.numel()
    params.height = 1
    params.pitch = 0
    params.value = 0
    return params


def _add_child_graph_node(graph, deps, dst):
    from cuda.bindings import runtime as cuda_runtime

    body = _check_cuda_bindings(cuda_runtime.cudaGraphCreate(0))
    _check_cuda_bindings(
        cuda_runtime.cudaGraphAddMemsetNode(body, [], 0, _memset_params(dst))
    )
    return _check_cuda_bindings(
        cuda_runtime.cudaGraphAddChildGraphNode(graph, deps, len(deps), body)
    )


def _add_conditional_node(graph, deps, dst):
    from cuda.bindings import driver, runtime as cuda_runtime

    handle = _check_cuda_bindings(
        cuda_runtime.cudaGraphConditionalHandleCreate(graph, 1, 1)
    )
    params = cuda_runtime.cudaGraphNodeParams()
    params.type = cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
    params.conditional.handle = handle
    params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
    params.conditional.size = 1
    node = _check_cuda_bindings(
        cuda_runtime.cudaGraphAddNode(graph, deps, None, len(deps), params)
    )
    _check_cuda_bindings(
        cuda_runtime.cudaGraphAddMemsetNode(
            params.conditional.phGraph_out[0], [], 0, _memset_params(dst)
        )
    )
    return node


_NESTED_NODE_ADDERS = {
    "child_graph": _add_child_graph_node,
    "conditional": _add_conditional_node,
}


# cuda.bindings is NVIDIA-only; graph annotation APIs have no ROCm equivalent.
@instantiate_parametrized_tests
@skipIfRocm
@requires_cuda
@requires_cuda_python_bindings
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestMarkKernels(TestCase):
    def setUp(self):
        super().setUp()
        # Annotations are enabled per-capture via torch.cuda.graph(..,
        # enable_annotations=True); there is no global toggle.
        _reset_kernel_annotations()
        self.addCleanup(_reset_kernel_annotations)

    def _count_phases(self, name):
        """Count (forward, backward) annotations recorded under ``name``."""
        fwd = bwd = 0
        for anns in get_kernel_annotations().values():
            for ann in anns:
                if ann.get("name") != name:
                    continue
                if ann.get("autograd_phase") == "backward":
                    bwd += 1
                else:
                    fwd += 1
        return fwd, bwd

    def _distinct_annotations(self):
        """The set of distinct resolved annotations, one per kernel group.

        Asserts every toolsId resolved to exactly one annotation dict, then
        normalizes kernels sharing an annotation down to one entry (a
        frozenset of the dict's items). This makes exact assertions possible
        while staying invariant over how many kernels each op decomposes
        into.
        """
        distinct = set()
        for tid, anns in get_kernel_annotations().items():
            self.assertEqual(len(anns), 1, f"toolsId {hex(tid)}: {anns}")
            distinct.add(frozenset(anns[0].items()))
        return distinct

    def assertAnnotations(self, *expected):
        """Assert the distinct resolved annotations equal ``expected`` dicts."""
        self.assertEqual(
            self._distinct_annotations(), {frozenset(e.items()) for e in expected}
        )

    def _bwd(self, ann):
        return {**ann, "autograd_phase": "backward"}

    def test_noop_outside_capture(self):
        x = torch.randn(8, device="cuda")
        with mark_kernels("test"):
            _ = x + 1
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_event_nodes_are_annotatable(self):
        """Event record/wait nodes are annotatable so the profiler can place their
        spans on the intended stream lane (they are stream-ordered but do not
        occupy the device)."""
        from cuda.bindings import driver

        annotatable = _get_annotatable_types()
        for name in (
            "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
            "CU_GRAPH_NODE_TYPE_WAIT_EVENT",
        ):
            self.assertIn(getattr(driver.CUgraphNodeType, name), annotatable)

    def test_host_nodes_are_annotatable(self):
        """Host nodes are annotatable so the profiler can place their spans on the
        intended stream lane (they are stream-ordered but do not occupy the
        device)."""
        from cuda.bindings import driver

        self.assertIn(
            driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_HOST, _get_annotatable_types()
        )

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
            self.assertEqual(annotations[tools_id], [{"name": "reduction"}])

    def test_single_scope_at_capture_start_uses_root_fallback(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            # mark_kernels is the first captured work here, so the entry
            # frontier is empty and the implementation must fall back to
            # newly created graph roots.
            with mark_kernels("phase_a"):
                _ = x + 1

        self.assertAnnotations({"name": "phase_a"})

    def test_multiple_scopes_no_overlap(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("scope_1"):
                _ = x + 1
            with mark_kernels("scope_2"):
                _ = x * 2

        # One annotation per toolsId (asserted by the helper) means no kernel
        # was claimed by both scopes.
        self.assertAnnotations({"name": "scope_1"}, {"name": "scope_2"})

    def test_dict_annotation(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        annotation = {"name": "all_gather", "Group size": 2, "dtype": "bfloat16"}
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels(annotation):
                _ = x + 1

        self.assertAnnotations(annotation)

    def test_clear_resets_state(self):
        """The deprecated public entry point still clears, and warns while doing it."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("test"):
                _ = x + 1

        self.assertGreater(len(get_kernel_annotations()), 0)
        with self.assertWarnsRegex(FutureWarning, "clear_kernel_annotations"):
            clear_kernel_annotations()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_failed_capture_end_discards_annotations(self):
        """A completed scope plus a failing capture_end used to strand entries keyed by
        the capture graph id. Nothing remaps them (that happens in instantiate) and
        remove_kernel_annotations matches exec ids, so they outlived the process."""
        x = torch.randn(8, device="cuda")
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        entry_stream = torch.cuda.current_stream()
        with self.assertRaises(Exception):
            with torch.cuda.graph(graph, enable_annotations=True):
                with mark_kernels("completed_scope"):
                    _ = x + 1
                side.wait_event(entry_stream.record_event())
                # Forked and never joined, so cudaStreamEndCapture fails.
                with torch.cuda.stream(side):
                    _ = x * 3

        self.assertEqual(len(get_kernel_annotations()), 0)
        # A failed capture_end must still unwind the stream context, or the graph's
        # private stream stays current and every later test inherits it.
        self.assertEqual(torch.cuda.current_stream(), entry_stream)

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

        self.assertAnnotations()

    def test_only_annotates_scope_kernels(self):
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            _ = x + 1
            _ = x * 2
            with mark_kernels("tagged"):
                _ = x + 3
            _ = x - 1

        self.assertAnnotations({"name": "tagged"})

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

        self.assertAnnotations({"name": "outer"}, {"name": "inner"})

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

        # Nested kernels: inner wins common keys ("name", "stream"),
        # inner-only keys preserved; nothing of outer's survives since
        # inner_ann covers all its keys.
        self.assertAnnotations(outer_ann, inner_ann)

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

        # Inner wins common keys ("name", "stream") and keeps its own;
        # outer contributes nothing since inner covers all its keys.
        self.assertAnnotations(inner_ann)

    def test_string_scope_in_dict_scope_contends_for_name(self):
        """A string annotation wraps to {"name": ...}, so a string scope
        nested inside a dict scope with a "name" key contends for the same
        slot; the inner scope wins. This pins the wrapped-string format and
        the merge ordering as observable pickle-format behavior."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "outer", "dtype": "bfloat16"}):
                with mark_kernels("inner"):
                    _ = x + 1

        # Inner string scope wins the "name" slot; outer-only keys survive.
        self.assertAnnotations({"name": "inner", "dtype": "bfloat16"})

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
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("auto"):
                _ = x + 1

        self.assertAnnotations({"name": "auto"})

    def test_enable_annotations_does_not_clear(self):
        """Annotations from a previous graph survive a second capture."""
        graph1 = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")

        with torch.cuda.graph(graph1, enable_annotations=True):
            with mark_kernels("first"):
                _ = x + 1

        self.assertAnnotations({"name": "first"})

        graph2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph2, enable_annotations=True):
            with mark_kernels("second"):
                _ = x * 2

        # Both graphs' annotations should be present.
        self.assertAnnotations({"name": "first"}, {"name": "second"})

    def test_enable_annotations_remaps_to_exec_graph(self):
        """enable_annotations=True must remap toolsIds to the exec graph ID."""
        from cuda.bindings import runtime as cuda_runtime

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

        self.assertAnnotations({"name": "aux", "stream": expected_stream_id})

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

        self.assertAnnotations({"name": "aux_branch", "stream": expected_stream_id})

    def test_mark_stream_does_not_mutate_caller_annotation(self):
        """The stream id goes onto a copy. The annotation is stored by reference, so
        writing it through would leak back to the caller and, for a dict reused across
        lanes, retag the regions already recorded with it to the last lane."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda")
        capture_stream = torch.cuda.Stream()
        lane_a = torch.cuda.Stream()
        lane_b = torch.cuda.Stream()
        id_a, id_b = _get_stream_id(lane_a), _get_stream_id(lane_b)
        shared = {"name": "comm"}

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=capture_stream, enable_annotations=True):
            for lane in (lane_a, lane_b):
                lane_ready = capture_stream.record_event()
                lane_done = torch.cuda.Event()
                with mark_stream(lane, shared):
                    lane.wait_event(lane_ready)
                    _ = x * 2
                    lane.record_event(lane_done)
                capture_stream.wait_event(lane_done)

        self.assertNotIn("stream", shared)
        self.assertAnnotations(
            {"name": "comm", "stream": id_a}, {"name": "comm", "stream": id_b}
        )

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
            graph_ids_by_name.setdefault(anns[0]["name"], set()).add(tools_id >> 32)

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

    def test_resolve_and_remap_sequence(self):
        """resolve_and_remap over a sequence keys each graph to its exec id.

        Uses default (keep_graph=False) graphs to show the capture id is
        recovered from the graph itself, not the (destroyed) graph template.
        The context manager already resolves and remaps on exit, so calling
        resolve_and_remap once per graph afterwards must be idempotent.
        """
        graphs = [torch.cuda.CUDAGraph() for _ in range(3)]
        x = torch.randn(8, device="cuda")

        for i, graph in enumerate(graphs):
            with torch.cuda.graph(graph, enable_annotations=True):
                with mark_kernels(f"graph_{i}"):
                    _ = x + i

        for graph in graphs:
            resolve_and_remap(graph)

        graph_ids_by_name: dict[str, set] = {}
        for tools_id, anns in get_kernel_annotations().items():
            self.assertEqual(len(anns), 1)
            graph_ids_by_name.setdefault(anns[0]["name"], set()).add(tools_id >> 32)

        for i, graph in enumerate(graphs):
            exec_id = self._exec_graph_id(graph)
            self.assertEqual(graph_ids_by_name[f"graph_{i}"], {exec_id})

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
        # Exactly the one mul kernel: the pre-existing dependent branch
        # (base * 2) must not be swept up by the scope's walk.
        self.assertEqual(len(annotations), 1)
        self.assertAnnotations({"name": "tagged"})

    def test_backward_kernels_annotated(self):
        """Backward kernels of ops run inside a scope carry its annotation."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("fwd_scope"):
                y = (x * 2).sin()
            z = y.sum()
            _ = torch.autograd.grad(z, x)

        # Forward kernels (mul, sin) tagged plain; backward kernels of their
        # autograd nodes (SinBackward0, MulBackward0), captured outside the
        # forward scope, tagged via node-creation hooks with
        # autograd_phase=backward. Nothing else (sum, grad plumbing) tagged.
        ann = {"name": "fwd_scope"}
        self.assertAnnotations(ann, self._bwd(ann))
        self.assertEqual(self._count_phases("fwd_scope")[0], 2)

    def test_backward_direct_with_preallocated_grad(self):
        """Direct .backward() with .grad preallocated before capture, so the
        AccumulateGrad write is an in-place add on a stable buffer (the
        capture-safe idiom) rather than an allocation."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("scope"):
                y = (x * 2).sin()
            y.sum().backward()

        # AccumulateGrad's in-place add stays untagged (helper would flag it).
        ann = {"name": "scope"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_multi_use_leaf(self):
        """A leaf used by two scopes: each scope's backward nodes carry their
        own scope's annotation."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("branch_a"):
                a = x.sin()
            with mark_kernels("branch_b"):
                b = x.cos()
            (a + b).sum().backward()

        # Each branch's forward and backward kernels carry exactly their own
        # scope; the shared add/sum/AccumulateGrad work carries neither.
        ann_a, ann_b = {"name": "branch_a"}, {"name": "branch_b"}
        self.assertAnnotations(ann_a, ann_b, self._bwd(ann_a), self._bwd(ann_b))

    def test_backward_false_annotates_forward_only(self):
        """backward=False tags the two forward kernels and nothing from the
        backward pass. The keyword's presence in the signature is also the
        documented probe for native backward support, so pin both."""
        import inspect

        self.assertIn("backward", inspect.signature(mark_kernels).parameters)

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("fwd_only", backward=False):
                y = (x * 2).sin()
            z = y.sum()
            _ = torch.autograd.grad(z, x)

        # Only the plain forward tag exists: no backward-phase variant at all.
        self.assertAnnotations({"name": "fwd_only"})
        self.assertEqual(self._count_phases("fwd_only")[0], 2)

    def test_backward_false_outer_keys_inherited_via_inner_scope(self):
        """A backward=False scope still contributes its annotation to the
        region, so nodes hooked by a nested backward=True scope freeze the
        outer keys into their backward tags."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "outer", "phase": "P"}, backward=False):
                with mark_kernels("inner"):
                    y = (x * 2).sin()
            _ = torch.autograd.grad(y.sum(), x)

        merged = {"name": "inner", "phase": "P"}
        self.assertAnnotations(merged, self._bwd(merged))

    def test_backward_scope_inside_custom_backward_wins(self):
        """A mark_kernels opened inside a node's backward execution is
        dynamically inner, so it outranks the node's inherited annotation."""

        class Scoped(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                with mark_kernels({"name": "bwd_inner", "who": "inner"}):
                    return grad_out * x.cos()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "fwd", "who": "outer", "fwd_only": 1}):
                y = Scoped.apply(x)
            _ = torch.autograd.grad(y.sum(), x)

        # Backward kernels: the inner scope wins common keys ("name", "who"),
        # keys unique to the inherited node annotation ("fwd_only") survive
        # the merge, and the kernels are marked as backward work.
        self.assertAnnotations(
            {"name": "fwd", "who": "outer", "fwd_only": 1},
            self._bwd({"name": "bwd_inner", "who": "inner", "fwd_only": 1}),
        )

    def test_backward_scope_around_backward_call_merges(self):
        """A mark_kernels active around backward() merges into backward
        kernels, losing common keys to each node's inherited annotation."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "fwd", "layer": 3}):
                y = (x * 2).sin()
            with mark_kernels({"name": "bwd_pass", "step": 7}, backward=False):
                _ = torch.autograd.grad(y.sum(), x)

        # Hooked-node backward kernels: the node annotation is dynamically
        # inner, so it wins "name" and keeps its forward keys, while
        # wrapper-only keys ("step") survive. Backward kernels of unmarked
        # nodes (sum's) get the wrapper tag alone.
        self.assertAnnotations(
            {"name": "fwd", "layer": 3},
            {"name": "fwd", "layer": 3, "step": 7, "autograd_phase": "backward"},
            {"name": "bwd_pass", "step": 7},
        )

    def test_backward_unmarked_nodes_not_tagged(self):
        """Backward kernels of nodes created outside any scope stay untagged:
        appending an unmarked op must not change the scope's tag counts."""
        counts = {}
        for with_unmarked in (False, True):
            _reset_kernel_annotations()
            graph = torch.cuda.CUDAGraph()
            x = torch.randn(8, device="cuda", requires_grad=True)

            with torch.cuda.graph(graph, enable_annotations=True):
                with mark_kernels("scope"):
                    y = x.sin()
                if with_unmarked:
                    y = y * 3  # unmarked following node
                _ = torch.autograd.grad(y.sum(), x)

            counts[with_unmarked] = self._count_phases("scope")

        # If MulBackward0 (of the unmarked y * 3) were wrongly tagged, the
        # backward count would grow; both runs must tag identically.
        self.assertEqual(counts[True], counts[False])
        fwd, bwd = counts[False]
        self.assertEqual(fwd, 1)
        self.assertGreaterEqual(bwd, 1)

    def test_backward_accumulate_grad_excluded(self):
        """AccumulateGrad work is never tagged: a .backward() run (which
        executes AccumulateGrad) tags exactly as many backward kernels as an
        autograd.grad run (which has no AccumulateGrad) of the same graph."""
        bwd_counts = {}
        for mode in ("grad", "backward"):
            _reset_kernel_annotations()
            graph = torch.cuda.CUDAGraph()
            x = torch.randn(8, device="cuda", requires_grad=True)
            x.grad = torch.zeros_like(x)

            with torch.cuda.graph(graph, enable_annotations=True):
                with mark_kernels("scope"):
                    y = (x * 2).sin()
                if mode == "backward":
                    y.sum().backward()
                else:
                    _ = torch.autograd.grad(y.sum(), x)

            fwd, bwd = self._count_phases("scope")
            self.assertEqual(fwd, 2, mode)
            self.assertGreaterEqual(bwd, 2, mode)
            bwd_counts[mode] = bwd

        self.assertEqual(bwd_counts["backward"], bwd_counts["grad"])

    def test_backward_shared_nonleaf_multi_consumer(self):
        """A shared non-leaf feeding two marked regions: gradient
        aggregation for the shared producer (InputBuffer, outside any
        node's brackets) must not be attributed to either region, so each
        region's counts match the single-consumer baseline."""
        counts = {}
        for consumers in (1, 2):
            _reset_kernel_annotations()
            graph = torch.cuda.CUDAGraph()
            x = torch.randn(8, device="cuda", requires_grad=True)

            with torch.cuda.graph(graph, enable_annotations=True):
                base = x * 2  # shared non-leaf producer, unmarked
                with mark_kernels("region_a"):
                    out = base.sin()
                if consumers == 2:
                    with mark_kernels("region_b"):
                        out = out + base.cos()
                _ = torch.autograd.grad(out.sum(), x)

            counts[consumers] = self._count_phases("region_a")

        # region_a's attribution must not absorb the shared producer's
        # gradient aggregation work that appears with the second consumer.
        self.assertEqual(counts[2], counts[1])
        fwd_b, bwd_b = self._count_phases("region_b")
        self.assertEqual(fwd_b, 2)  # cos + add
        self.assertGreaterEqual(bwd_b, 1)

    def test_backward_nested_backward_inside_custom_backward(self):
        """A custom backward that itself calls torch.autograd.grad: nodes
        of the inner graph built there are owned by the scope opened inside
        the custom backward."""

        class InnerGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                with mark_kernels("inner_bwd"):
                    with torch.enable_grad():
                        t = x.detach().requires_grad_()
                        z = (t * 3).cos()
                        (gt,) = torch.autograd.grad(z.sum(), t)
                return grad_out * gt

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("fwd"):
                y = InnerGrad.apply(x)
            _ = torch.autograd.grad(y.sum(), x)

        # "inner_bwd" owns the inner graph's work as backward (its lexical
        # kernels run inside InnerGrad's bracket, whose posthook tag wins the
        # per-kernel merge over the scope's own plain record); "fwd" owns its
        # forward kernel and the rest of InnerGrad's backward.
        self.assertAnnotations(
            {"name": "fwd"},
            self._bwd({"name": "fwd"}),
            self._bwd({"name": "inner_bwd"}),
        )

    def test_backward_mark_stream_differentiable(self):
        """mark_stream around differentiable work: the stream id recorded
        for forward lane assignment also propagates into backward tags."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        side = torch.cuda.Stream()

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=capture_stream, enable_annotations=True):
            fork = torch.cuda.Event()
            fork = capture_stream.record_event()
            side.wait_event(fork)
            with mark_stream(side, "lane"):
                y = (x * 2).sin()
            capture_stream.wait_stream(side)
            _ = torch.autograd.grad(y.sum(), x)

        ann = {"name": "lane", "stream": _get_stream_id(side)}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_nested_dict_annotations_without_name(self):
        """Nested dict scopes need no "name" key; backward merge still
        applies inner-wins on common keys and unions the rest."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"layer": 1, "kind": "outer"}):
                with mark_kernels({"kind": "inner", "extra": True}):
                    y = (x * 2).sin()
            _ = torch.autograd.grad(y.sum(), x)

        merged = {"layer": 1, "kind": "inner", "extra": True}
        self.assertAnnotations(merged, self._bwd(merged))

    def test_backward_node_after_inner_scope_exit(self):
        """A node created after an inner scope exits, while the outer is
        still open, inherits only the outer annotation."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("outer"):
                with mark_kernels("inner"):
                    y = x.sin()
                z = y * 3  # created under outer only
            _ = torch.autograd.grad(z.sum(), x)

        # sin: forward tagged inner (inner wins merge), backward inherits
        # both (inner wins). mul: forward and backward tagged outer only.
        self.assertAnnotations(
            {"name": "inner"},
            {"name": "outer"},
            self._bwd({"name": "inner"}),
            self._bwd({"name": "outer"}),
        )

    def test_backward_explicit_backward_entry_point(self):
        """torch.autograd.backward(...) (not Tensor.backward) also runs the
        brackets."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("scope"):
                y = (x * 2).sin()
            torch.autograd.backward((y,), (torch.ones_like(y),))

        ann = {"name": "scope"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_create_graph_second_order_attributed(self):
        """Nodes created during a create_graph backward inherit the forward
        scope's annotations, so a second-order backward is attributed too."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("L"):
                y = (x * 2).sin()
            (gx,) = torch.autograd.grad(y.sum(), x, create_graph=True)
            _ = torch.autograd.grad(gx.sum(), x)

        ann = {"name": "L"}
        self.assertAnnotations(ann, self._bwd(ann))
        # Second-order kernels tag identically to first-order ones, so the
        # distinct set can't tell them apart; the backward count can:
        # first-order kernels (>=2) plus second-order kernels from nodes
        # created during the first backward.
        self.assertGreaterEqual(self._count_phases("L")[1], 4)

    def test_backward_second_order_through_uncaptured_first(self):
        """Ownership propagates even when the first backward runs eagerly:
        only the second-order backward is captured, and it is attributed."""
        x = torch.randn(8, device="cuda", requires_grad=True)
        graph_fwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_fwd, enable_annotations=True):
            with mark_kernels("L"):
                y = (x * 2).sin()

        (gx,) = torch.autograd.grad(y.sum(), x, create_graph=True)

        graph_bwd2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_bwd2, enable_annotations=True):
            _ = torch.autograd.grad(gx.sum(), x)

        ann = {"name": "L"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_checkpoint_recompute_attributed(self):
        """Checkpoint recomputation during backward creates fresh nodes;
        scopes inside the recomputed forward still own the recompute work."""

        def fn(x):
            with mark_kernels("ckpt"):
                return (x * 2).sin()

        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)
        with torch.cuda.graph(graph, enable_annotations=True):
            y = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
            _ = torch.autograd.grad(y.sum(), x)

        # The scope tags its original forward kernels plain; the recomputed
        # forward kernels (created during backward) and the backward kernels
        # of the recomputed nodes get the backward-phase tag.
        ann = {"name": "ckpt"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_make_graphed_callables_annotations(self):
        """make_graphed_callables(enable_annotations=True) records marks from
        inside the module for both its forward and backward captures, and the
        graphed module still produces input and parameter gradients."""

        class Marked(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8)

            def forward(self, x):
                with mark_kernels("layer"):
                    return self.lin(x).relu()

        module = Marked().cuda()
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        graphed = torch.cuda.make_graphed_callables(
            module, (x,), enable_annotations=True
        )
        y = graphed(x)
        y.sum().backward()
        torch.cuda.synchronize()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(module.lin.weight.grad)
        self.assertIsNotNone(module.lin.bias.grad)

        ann = {"name": "layer"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_captured_in_separate_graph(self):
        """Backward captured in its own graph (the make_graphed_callables
        pattern) still tags its kernels with the forward scope's annotation."""
        x = torch.randn(8, device="cuda", requires_grad=True)
        graph_fwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_fwd, enable_annotations=True):
            with mark_kernels("layer"):
                y = (x * 2).sin()

        graph_bwd = torch.cuda.CUDAGraph()
        grad_out = torch.ones_like(y)
        with torch.cuda.graph(graph_bwd, enable_annotations=True):
            _ = torch.autograd.grad(y, x, grad_outputs=grad_out, retain_graph=True)

        ann = {"name": "layer"}
        self.assertAnnotations(ann, self._bwd(ann))

    def test_backward_nested_scopes_inner_wins(self):
        """Backward kernels of nested scopes merge with the inner scope
        winning common keys, matching the forward merge order."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "outer", "tag": "O"}):
                with mark_kernels("inner"):
                    y = (x * 2).sin()
            z = y.sum()
            _ = torch.autograd.grad(z, x)

        merged = {"name": "inner", "tag": "O"}
        self.assertAnnotations(merged, self._bwd(merged))

    def test_backward_outside_capture_is_noop(self):
        """Hooks registered by a captured forward are inert in eager backward."""
        graph = torch.cuda.CUDAGraph()
        x = torch.randn(8, device="cuda", requires_grad=True)

        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels("scope"):
                y = (x * 2).sum()

        before = {tid: list(anns) for tid, anns in get_kernel_annotations().items()}
        # Eager replay-independent backward: not capturing, hooks must no-op.
        y.backward()
        torch.cuda.synchronize()
        resolve_pending_annotations()
        self.assertEqual(dict(get_kernel_annotations()), before)

    def test_eager_forward_backward_noop(self):
        x = torch.randn(8, device="cuda", requires_grad=True)
        with mark_kernels("eager"):
            y = (x * 2).sum()
        y.backward()
        torch.cuda.synchronize()
        self.assertEqual(len(get_kernel_annotations()), 0)

    def test_scope_inside_conditional_body_records_nothing(self):
        """A scope inside a torch.cond body warns and records no annotations.

        begin_capture_to_if_node captures into a separate cudaGraph_t, so node ids
        there are in the body graph's id space; remap_to_exec_graph only rekeys the
        top-level capture id, so anything recorded would be a key matching nothing.
        """
        from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

        x = torch.ones([2000], device="cuda")
        pred = torch.tensor(True, device="cuda")
        g = torch.cuda.CUDAGraph(keep_graph=True)

        with torch.cuda.graph(g, enable_annotations=True):
            with mark_kernels("outer_region"):
                z = x + 1
            with self.assertWarnsRegex(UserWarning, "conditional-node body"):
                with _if_body(pred):
                    with mark_kernels("inside_body"):
                        _ = z * 2
        g.instantiate()

        # Only the top-level scope is recorded, and its key matches the exec graph.
        annotations = get_kernel_annotations()
        self.assertEqual(
            [a for anns in annotations.values() for a in anns],
            [{"name": "outer_region"}],
        )
        with self.assertWarns(UserWarning):  # the graph has a conditional node
            exec_graph_id = g.get_graph_data()["exec_graph_id"]
        for tools_id in annotations:
            self.assertEqual(tools_id >> 32, exec_graph_id)

    @parametrize("kind", ["child_graph", "conditional"])
    def test_nested_graph_node_in_scope_warns(self, kind):
        """A scope containing a nested graph node warns; the rest is annotated.

        The dependent-edge walk stops at such a node, so the work in its body is
        left unannotated (and its ids, being in the body graph's id space, would
        never be rekeyed by remap_to_exec_graph). What is recorded stays correct.
        """
        from cuda.bindings import runtime as cuda_runtime

        dst = torch.zeros(64, device="cuda")
        x = torch.zeros([2000], device="cuda")
        g = torch.cuda.CUDAGraph(keep_graph=True)

        with self.assertWarnsRegex(UserWarning, "left unannotated"):
            with torch.cuda.graph(
                g, enable_annotations=True, capture_error_mode="relaxed"
            ):
                with mark_kernels("region"):
                    y = x + 1
                    stream = torch.cuda.current_stream().cuda_stream
                    _s, _i, cap_graph, deps, _e, num = _check_cuda_bindings(
                        cuda_runtime.cudaStreamGetCaptureInfo(stream)
                    )
                    _NESTED_NODE_ADDERS[kind](cap_graph, list(deps[:num]), dst)
                    _ = y * 2

        # The two elementwise kernels in the scope are still annotated correctly.
        annotations = get_kernel_annotations()
        self.assertEqual(len(annotations), 2)
        for anns in annotations.values():
            self.assertEqual(anns, [{"name": "region"}])


# cuda.bindings is NVIDIA-only; get_graph_data has no ROCm equivalent.
@instantiate_parametrized_tests
@skipIfRocm
@requires_cuda
@requires_cuda_python_bindings
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
class TestGetGraphData(TestCase):
    def test_batch_mem_op_node(self):
        from cuda.bindings import driver

        device = _check_cuda_bindings_driver(
            driver.cuDeviceGet(torch.cuda.current_device())
        )
        can_use_stream_mem_ops = _check_cuda_bindings_driver(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
                device,
            )
        )
        if not can_use_stream_mem_ops:
            self.skipTest("device does not support stream memory operations")

        value = torch.zeros(1, dtype=torch.int32, device="cuda")
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            _check_cuda_bindings_driver(
                driver.cuStreamWriteValue32(
                    torch.cuda.current_stream().cuda_stream,
                    value.data_ptr(),
                    1,
                    driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
            )

        g.instantiate()
        data = g.get_graph_data()
        batch_mem_op_nodes = [
            node for node in data["nodes"] if node["node_type"] == "batch_mem_op"
        ]
        self.assertEqual(len(batch_mem_op_nodes), 1)

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
            self.assertIn("grid_dim", node)
            self.assertIn("block_dim", node)
            self.assertIn("dependencies", node)
            self.assertIn("dependents", node)
            self.assertEqual(node["graph_id"], exec_graph_id)
            self.assertEqual(node["tools_id"], (exec_graph_id << 32) | node["node_id"])
            if node["node_type"] == "kernel":
                for dims in (node["grid_dim"], node["block_dim"]):
                    self.assertIsInstance(dims, tuple)
                    self.assertEqual(len(dims), 3)
                    for d in dims:
                        self.assertIsInstance(d, int)
                        self.assertGreater(d, 0)
            else:
                self.assertIsNone(node["grid_dim"])
                self.assertIsNone(node["block_dim"])

        kernel_nodes = [n for n in data["nodes"] if n["node_type"] == "kernel"]
        self.assertGreater(len(kernel_nodes), 0)
        for kn in kernel_nodes:
            self.assertIsNotNone(kn["kernel_name"])
            self.assertIsInstance(kn["kernel_name"], str)

        self.assertIn("edges", data)
        n_nodes = len(data["nodes"])
        for edge in data["edges"]:
            for key in ("from", "to", "from_port", "to_port", "type"):
                self.assertIn(key, edge)
                self.assertIsInstance(edge[key], int)
            self.assertGreaterEqual(edge["from"], 0)
            self.assertLess(edge["from"], n_nodes)
            self.assertGreaterEqual(edge["to"], 0)
            self.assertLess(edge["to"], n_nodes)
            # A plain capture only produces ordinary full-serialization edges.
            self.assertEqual(edge["from_port"], 0)
            self.assertEqual(edge["to_port"], 0)
            self.assertEqual(edge["type"], 0)
        # The edge list and the per-node lists describe the same relation.
        self.assertEqual(
            {(e["from"], e["to"]) for e in data["edges"]},
            {
                (dep, node["index"])
                for node in data["nodes"]
                for dep in node["dependencies"]
            },
        )

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

    def test_cached_across_instantiate_hooks(self):
        # Within one instantiate(), every post-instantiate hook that calls
        # get_graph_data() shares a single query (same object); the cache is
        # dropped afterwards so a later call recomputes a fresh dict.
        g = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.zeros([2000], device="cuda")
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            _ = (x + 1).relu()

        seen = []
        g.register_post_instantiate_hook(lambda cg: seen.append(cg.get_graph_data()))
        g.register_post_instantiate_hook(lambda cg: seen.append(cg.get_graph_data()))
        g.instantiate()

        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])

        after = g.get_graph_data()
        self.assertIsNot(after, seen[0])
        self.assertEqual(after, seen[0])

    @parametrize("kind", ["child_graph", "conditional"])
    def test_nested_graph_node_warns_but_reports_top_level(self, kind):
        """A nested cudaGraph_t warns; the top-level nodes are still reported.

        cudaGraphGetNodes does not descend into a child graph or a conditional
        body, so that work is missing from the result -- but the ids that are
        reported stay valid, since the exec graph preserves top-level node ids.
        """
        dst = torch.zeros(64, device="cuda")
        x = torch.zeros([2000], device="cuda")
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            _ = x + 1
        _NESTED_NODE_ADDERS[kind](g.raw_cuda_graph(), [], dst)
        g.instantiate()

        with self.assertWarnsRegex(UserWarning, kind):
            data = g.get_graph_data()

        # The nested node itself is reported, its body is not: exactly one memset
        # lives in the body, and it must not appear among the returned nodes.
        self.assertEqual(len([n for n in data["nodes"] if n["node_type"] == kind]), 1)
        self.assertEqual(
            len([n for n in data["nodes"] if n["node_type"] == "memset"]), 0
        )
        exec_graph_id = data["exec_graph_id"]
        for node in data["nodes"]:
            self.assertEqual(node["tools_id"], (exec_graph_id << 32) | node["node_id"])

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


# Runs everywhere (no capture): the public probe must agree with the private
# gate that mark_kernels no-ops on, whatever this machine supports.
@instantiate_parametrized_tests
class TestIsAvailable(TestCase):
    def test_matches_private_gate(self):
        expected = torch.cuda.is_available() and not _is_tools_id_unavailable()
        self.assertEqual(is_available(), expected)

    @parametrize(
        "err_name",
        ["cudaErrorInsufficientDriver", "cudaErrorCallRequiresNewerDriver"],
    )
    def test_probe_rejects_unusable_driver(self, err_name):
        """Only an error that proves the API is there counts as supported. ROCm answers
        cudaErrorInsufficientDriver, which used to read as supported and then blew up on
        the first real call."""
        import torch.cuda._graph_annotations as ga

        if ga._cuda_runtime is None:
            self.skipTest("cuda-bindings not installed")
        err = getattr(ga._cuda_runtime.cudaError_t, err_name)
        with unittest.mock.patch.object(
            ga._cuda_runtime, "cudaGraphNodeGetToolsId", lambda _node: (err, 0)
        ):
            self.assertFalse(ga._probe_tools_id())

    def test_rocm_build_reports_bindings_absent(self):
        """cuda-bindings installs and imports fine on a ROCm box but cannot work there,
        so the shared gate reports it absent rather than letting each caller discover
        that at the first call (cudaErrorInsufficientDriver)."""
        import importlib

        import torch.cuda._utils as u

        orig = torch.version.hip
        try:
            torch.version.hip = "6.0.0"
            importlib.reload(u)
            self.assertFalse(u._HAS_CUDA_BINDINGS)
            self.assertIsNone(u._cuda_bindings_runtime)
            self.assertIsNone(u._cuda_bindings_driver)
        finally:
            torch.version.hip = orig
            importlib.reload(u)

    def test_probe_accepts_invalid_value(self):
        # A driver that has the API rejects the null node with cudaErrorInvalidValue.
        import torch.cuda._graph_annotations as ga

        if ga._cuda_runtime is None:
            self.skipTest("cuda-bindings not installed")
        err = ga._cuda_runtime.cudaError_t.cudaErrorInvalidValue
        with unittest.mock.patch.object(
            ga._cuda_runtime, "cudaGraphNodeGetToolsId", lambda _node: (err, 0)
        ):
            self.assertTrue(ga._probe_tools_id())


# Host-side test of the graph-instantiate hook registry: consumers register hooks that a
# CUDAGraph fans out to at the end of instantiate(). The registry just passes the graph
# through, so a sentinel stands in for it. No CUDA needed.
class TestGraphInstantiateHooks(TestCase):
    def tearDown(self):
        import torch.cuda.graphs as cg

        cg._global_instantiate_hooks.clear()
        super().tearDown()

    @staticmethod
    def _run(graph):
        import torch.cuda.graphs as cg

        cg._run_global_hooks(cg._global_instantiate_hooks, graph)

    def test_register_run_unregister(self):
        import torch.cuda.graphs as cg

        seen = []
        graph = object()
        handle = cg.register_graph_instantiate_hook(lambda g: seen.append(g))
        self._run(graph)
        self.assertEqual(seen, [graph])
        handle.remove()
        self._run(graph)  # unregistered: not called again
        self.assertEqual(seen, [graph])

    def test_multiple_hooks_all_run(self):
        import torch.cuda.graphs as cg

        seen_a, seen_b = [], []
        cg.register_graph_instantiate_hook(lambda g: seen_a.append(g))
        cg.register_graph_instantiate_hook(lambda g: seen_b.append(g))
        graph = object()
        self._run(graph)
        self.assertEqual((seen_a, seen_b), ([graph], [graph]))

    def test_run_swallows_hook_errors(self):
        import torch.cuda.graphs as cg

        seen = []

        def boom(g):
            raise RuntimeError("boom")

        cg.register_graph_instantiate_hook(boom)
        cg.register_graph_instantiate_hook(lambda g: seen.append(g))
        graph = object()
        self._run(graph)  # first raises, second still runs
        self.assertEqual(seen, [graph])

    @requires_cuda
    def test_fires_once_on_real_graph_instantiate(self):
        import torch.cuda.graphs as cg

        seen = []
        handle = cg.register_graph_instantiate_hook(lambda gr: seen.append(gr))
        self.addCleanup(handle.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = x + 1

        # instantiate() ran once at capture_end and fired the hook with this graph.
        self.assertEqual(seen, [g])
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        # replay does not re-instantiate, so the hook is not called again.
        self.assertEqual(seen, [g])
        g.reset()


# Host-side test of the graph-destroy hook registry: consumers register per-resolver
# cleanup hooks that a CUDAGraph invokes (gated on _graph_destroy_hooks_active) via
# _run_graph_destroy_hooks when a CUDA graph is destroyed. No CUDA needed.
class TestGraphDestroyHooks(TestCase):
    def tearDown(self):
        import torch.cuda.graphs as cg

        cg._global_destroy_hooks.clear()
        super().tearDown()

    def test_register_run_unregister(self):
        import torch.cuda.graphs as cg

        seen = []
        self.assertFalse(cg._graph_destroy_hooks_active())
        handle = cg.register_graph_destroy_hook(lambda ids: seen.append(set(ids)))
        self.assertTrue(cg._graph_destroy_hooks_active())
        cg._run_graph_destroy_hooks({7})
        self.assertEqual(seen, [{7}])
        handle.remove()
        self.assertFalse(cg._graph_destroy_hooks_active())
        cg._run_graph_destroy_hooks({8})  # unregistered: not called again
        self.assertEqual(seen, [{7}])

    def test_multiple_hooks_all_run(self):
        import torch.cuda.graphs as cg

        seen_a, seen_b = [], []
        cg.register_graph_destroy_hook(lambda ids: seen_a.append(set(ids)))
        cg.register_graph_destroy_hook(lambda ids: seen_b.append(set(ids)))
        cg._run_graph_destroy_hooks({5})
        self.assertEqual((seen_a, seen_b), ([{5}], [{5}]))

    def test_run_swallows_hook_errors(self):
        import torch.cuda.graphs as cg

        seen = []

        def boom(ids):
            raise RuntimeError("boom")

        cg.register_graph_destroy_hook(boom)
        cg.register_graph_destroy_hook(lambda ids: seen.append(set(ids)))
        cg._run_graph_destroy_hooks({1})  # first raises, second still runs
        self.assertEqual(seen, [{1}])

    @requires_cuda
    def test_fires_with_recorded_exec_ids_on_teardown(self):
        import torch.cuda.graphs as cg

        exec_id = 0xABCD
        seen = []
        # Stand in for a consumer that records per-graph state at instantiate: stamp a
        # known exec id onto the graph so teardown has something to hand the destroy hook.
        inst = cg.register_graph_instantiate_hook(
            lambda gr: gr._recorded_exec_ids.add(exec_id)
        )
        dh = cg.register_graph_destroy_hook(lambda ids: seen.append(set(ids)))
        self.addCleanup(inst.remove)
        self.addCleanup(dh.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = x + 1
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        self.assertEqual(seen, [])  # not torn down yet
        # reset() tears down this capture cycle and fires the armed destroy callback,
        # handing the destroy hook the exec ids recorded on the graph.
        g.reset()
        self.assertEqual(seen, [{exec_id}])


# Pure registry-lifecycle logic, no CUDA needed. Seeds the module-level kernel
# annotation map directly and checks that remove_kernel_annotations purges only the
# requested exec graph ids (tools_id >> 32). (The graph dependency map lives on the
# profiler observer, not the module, and is exercised in the Cuspy suite.)
class TestRemoveKernelAnnotations(TestCase):
    @staticmethod
    def _tools_id(graph_id, node_id):
        return (graph_id << 32) | node_id

    def setUp(self):
        import torch.cuda._graph_annotations as ga

        super().setUp()
        ga._reset_kernel_annotations()
        self.addCleanup(ga._reset_kernel_annotations)

    def test_removes_only_requested_exec_id(self):
        import torch.cuda._graph_annotations as ga

        exec_a, exec_b = 1, 2
        ga.record_node_annotation(self._tools_id(exec_a, 10), {"name": "a"})
        ga.record_node_annotation(self._tools_id(exec_b, 10), {"name": "b"})

        ga.remove_kernel_annotations([exec_a])

        self.assertEqual(
            dict(ga.get_kernel_annotations()),
            {self._tools_id(exec_b, 10): [{"name": "b"}]},
        )

    def test_missing_and_empty_ids_are_noops(self):
        import torch.cuda._graph_annotations as ga

        ga.record_node_annotation(self._tools_id(2, 10), {"name": "b"})
        ga.remove_kernel_annotations([])  # empty: no-op
        ga.remove_kernel_annotations([99])  # unknown exec id: no-op
        self.assertEqual(
            dict(ga.get_kernel_annotations()), {self._tools_id(2, 10): [{"name": "b"}]}
        )


# Pure registry logic, no CUDA needed: the store keeps one merged annotation per node,
# which the public mapping wraps in a one-element list.
class TestAnnotationStore(TestCase):
    def setUp(self):
        import torch.cuda._graph_annotations as ga

        super().setUp()
        ga._reset_kernel_annotations()
        self.addCleanup(ga._reset_kernel_annotations)

    def test_writes_to_one_node_merge_first_wins(self):
        import torch.cuda._graph_annotations as ga

        # Scopes reach a node innermost first, so the first write keeps the shared keys.
        ga.record_node_annotation(7, {"name": "inner", "only_inner": 1})
        ga.record_node_annotation(7, {"name": "outer", "only_outer": 2})
        merged = {"name": "inner", "only_inner": 1, "only_outer": 2}
        self.assertEqual(ga.annotation_for(7), merged)
        self.assertEqual(dict(ga.get_kernel_annotations()), {7: [merged]})

    def test_view_is_live_and_read_only(self):
        import torch.cuda._graph_annotations as ga

        view = ga.get_kernel_annotations()
        self.assertEqual(len(view), 0)
        ga.record_node_annotation(7, {"name": "a"})
        self.assertEqual(dict(view), {7: [{"name": "a"}]})
        with self.assertRaises(TypeError):
            view[7] = [{"name": "b"}]  # type: ignore[index]


# Every global lifecycle registry behaves the same way -- register / fan out / remove, with
# per-hook errors swallowed so one consumer cannot break the step for another -- so the
# contract is checked once per registry. The instantiate and destroy registries have their own
# classes above; these are the four that pair with a per-graph hook.
GLOBAL_HOOK_REGISTRIES = [
    "capture_start",
    "capture_end",
    "instantiate",
    "replay_start",
    "replay_end",
]


@instantiate_parametrized_tests
class TestGraphGlobalLifecycleHooks(TestCase):
    def tearDown(self):
        import torch.cuda.graphs as cg

        for name in GLOBAL_HOOK_REGISTRIES:
            getattr(cg, f"_global_{name}_hooks").clear()
        super().tearDown()

    @staticmethod
    def _api(name):
        # The fan-out is private -- only CUDAGraph calls it -- so drive it the way the graph
        # does, with the registry the register_* function writes to.
        import torch.cuda.graphs as cg

        registry = getattr(cg, f"_global_{name}_hooks")
        return (
            getattr(cg, f"register_graph_{name}_hook"),
            lambda graph: cg._run_global_hooks(registry, graph),
        )

    @parametrize("name", GLOBAL_HOOK_REGISTRIES)
    def test_register_run_unregister(self, name):
        register, run = self._api(name)
        seen = []
        graph = object()  # the registry passes the graph through untouched
        handle = register(seen.append)
        run(graph)
        self.assertEqual(seen, [graph])
        handle.remove()
        run(graph)  # unregistered: not called again
        self.assertEqual(seen, [graph])

    @parametrize("name", GLOBAL_HOOK_REGISTRIES)
    def test_run_swallows_hook_errors(self, name):
        register, run = self._api(name)
        seen = []

        def boom(g):
            raise RuntimeError("boom")

        register(boom)
        register(seen.append)
        graph = object()
        run(graph)  # first raises, second still runs
        self.assertEqual(seen, [graph])

    @parametrize("name", GLOBAL_HOOK_REGISTRIES)
    def test_only_registration_is_public(self, name):
        import torch.cuda.graphs as cg

        self.assertIn(f"register_graph_{name}_hook", cg.__all__)
        self.assertFalse(hasattr(cg, f"run_graph_{name}_hooks"))

    @requires_cuda
    def test_fire_points_over_a_real_graph(self):
        # Each registry fires at its own point in one graph's life, for a graph the consumer
        # never built: capture start once, capture end once, instantiate once, and the replay
        # pair once per replay.
        import torch.cuda.graphs as cg

        seen = []
        for name in GLOBAL_HOOK_REGISTRIES:
            handle = getattr(cg, f"register_graph_{name}_hook")(
                lambda g, name=name: seen.append(name)
            )
            self.addCleanup(handle.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = x + 1
        # keep_graph=False instantiates from capture_end, so instantiate lands after it.
        self.assertEqual(seen, ["capture_start", "capture_end", "instantiate"])
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        self.assertEqual(
            seen[3:],
            ["replay_start", "replay_end", "replay_start", "replay_end"],
        )
        g.reset()

    @requires_cuda
    def test_global_hooks_run_before_per_graph_hooks(self):
        # Ordering matches instantiate(): the global fan-out runs first, then the graph's own
        # hooks. A consumer registering both sees them in that order.
        import torch.cuda.graphs as cg

        order = []
        handle = cg.register_graph_capture_start_hook(lambda g: order.append("global"))
        self.addCleanup(handle.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        g.register_capture_start_hook(lambda gr: order.append("per-graph"))
        with torch.cuda.graph(g):
            _ = x + 1
        self.assertEqual(order, ["global", "per-graph"])
        g.reset()

    @requires_cuda
    def test_capture_start_hook_sees_live_capture(self):
        # The hook fires with capture already under way -- that is what lets it read capture
        # state, and why its docs forbid issuing CUDA work.
        import torch.cuda.graphs as cg

        capturing = []
        handle = cg.register_graph_capture_start_hook(
            lambda g: capturing.append(torch.cuda.is_current_stream_capturing())
        )
        self.addCleanup(handle.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = x + 1
        self.assertEqual(capturing, [True])
        g.reset()

    @requires_cuda
    def test_replay_end_hook_fires_when_launch_raises(self):
        # The end hook is in a finally, so a start hook is always balanced by an end even if
        # the launch itself fails; the launch error still propagates.
        import torch.cuda.graphs as cg

        seen = []
        for name in ("replay_start", "replay_end"):
            handle = getattr(cg, f"register_graph_{name}_hook")(
                lambda g, name=name: seen.append(name)
            )
            self.addCleanup(handle.remove)

        x = torch.zeros(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = x + 1
        with unittest.mock.patch.object(
            torch._C._CUDAGraph, "replay", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                g.replay()
        self.assertEqual(seen, ["replay_start", "replay_end"])
        g.reset()


def _cupti_backend_available():
    """Whether the CUPTI annotation backend can be exercised: cupti-python present and
    Cuspy able to subscribe. Probed by actually bringing an observer up, since
    ``has_live_subscription`` is false until something holds a subscription."""
    try:
        from torch.profiler._cuspy.observers.node_timer import NodeTimerObserver
    except ImportError:
        return False
    try:
        return NodeTimerObserver().available
    except Exception:
        return False


@skipIfRocm
@requires_cuda
@requires_cuda_python_bindings
@unittest.skipIf(
    _is_tools_id_unavailable(),
    "cudaGraphNodeGetToolsId not available (needs cuda-compat >= 13.1)",
)
@unittest.skipIf(not _cupti_backend_available(), "requires Cuspy able to subscribe")
class TestCuptiAnnotationBackend(TestCase):
    """``annotation_config={"backend": "cupti"}``: nodes are attributed as CUPTI reports their creation
    rather than by walking the capture graph's dependent edges."""

    def setUp(self):
        _reset_kernel_annotations()
        self.addCleanup(_reset_kernel_annotations)
        # Attributing a node needs the mark_kernels scope open on the creating thread.
        ctx = torch.autograd.grad_mode.set_multithreading_enabled(False)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        # Holding an observer keeps Cuspy subscribed, which is what makes the CUPTI
        # backend selectable (and what "auto" probes for).
        from torch.profiler._cuspy.observers.node_timer import NodeTimerObserver

        self.observer = NodeTimerObserver()
        close = getattr(self.observer, "close", None)
        if close is not None:
            self.addCleanup(close)

    @staticmethod
    def _warm(x):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                x = x + 1
        torch.cuda.current_stream().wait_stream(s)
        return x

    def _annotations(self):
        return dict(get_kernel_annotations())

    def test_parity_with_edge_walk(self):
        # Both backends must attribute the same nodes the same way, and key them the same
        # way: the node ids must match, and every key must be rekeyed to that capture's exec
        # graph id (what lets an annotation join a trace's "graph node id"). Comparing keys
        # rather than just counts is what catches the CUPTI handler recording into a
        # different id space. The backward pass is part of the comparison because it is
        # attributed by a different mechanism -- hooks the scope installs on the autograd
        # nodes its forward creates -- that both backends share.
        from cuda.bindings import runtime as cuda_runtime

        def warm(x):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    torch.autograd.grad((x * 2).sin().sum(), x)
            torch.cuda.current_stream().wait_stream(s)

        by_backend = {}
        for backend in ("edge_walk", "cupti"):
            _reset_kernel_annotations()
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            warm(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                g, enable_annotations=True, annotation_config={"backend": backend}
            ):
                with mark_kernels("phase"):
                    y = (x * 2).sin()
                with mark_kernels({"name": "bwd_region", "lane": 1}):
                    torch.autograd.grad(y.sum(), x)
            exec_graph_id = _check_cuda_bindings(
                cuda_runtime.cudaGraphExecGetId(g.raw_cuda_graph_exec())
            )
            annotations = self._annotations()
            for tools_id in annotations:
                self.assertEqual(tools_id >> 32, exec_graph_id)
            by_backend[backend] = {
                tools_id & 0xFFFFFFFF: entries
                for tools_id, entries in annotations.items()
            }
        self.assertEqual(by_backend["cupti"], by_backend["edge_walk"])

        # Parity alone would also hold if both backends broke the same way, so pin the
        # annotation the hooks are there to produce: the forward scope owns its backward
        # kernels even though a different scope is open around the backward call, whose
        # other keys still merge in.
        self.assertIn(
            [{"name": "phase", "lane": 1, "autograd_phase": "backward"}],
            by_backend["cupti"].values(),
        )

    def test_scope_entered_before_stream_joins_capture(self):
        # The edge walk snapshots the CURRENT stream's capture state on scope entry, so a
        # scope entered while that stream is not yet capturing records nothing at all --
        # even though work inside it is captured. CUPTI reports each node as it is created,
        # so it is unaffected.
        def run(backend):
            _reset_kernel_annotations()
            x = self._warm(torch.randn(64, 64, device="cuda"))
            g = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            with torch.cuda.graph(
                g, enable_annotations=True, annotation_config={"backend": backend}
            ):
                capturing = torch.cuda.current_stream()
                joined = torch.cuda.Event()
                joined.record(capturing)
                with torch.cuda.stream(side):
                    with mark_kernels("region"):
                        # side is not capturing yet at scope entry; it joins here.
                        side.wait_event(joined)
                        for _ in range(3):
                            x = x + 1
                    done = torch.cuda.Event()
                    done.record(side)
                capturing.wait_event(done)
            return len(self._annotations())

        self.assertEqual(run("edge_walk"), 0)
        self.assertEqual(run("cupti"), 3)

    def test_nested_scopes_merge_inner_wins(self):
        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g, enable_annotations=True, annotation_config={"backend": "cupti"}
        ):
            with mark_kernels({"name": "outer", "shared": "outer", "only_outer": 1}):
                x = x + 1
                with mark_kernels({"name": "inner", "shared": "inner"}):
                    x = x + 1
        annotations = sorted(self._annotations().items())
        self.assertEqual(len(annotations), 2)
        self.assertEqual(
            annotations[0][1], [{"name": "outer", "shared": "outer", "only_outer": 1}]
        )
        # Inner wins the keys both scopes set; the outer-only key still comes through.
        self.assertEqual(
            annotations[1][1],
            [{"name": "inner", "shared": "inner", "only_outer": 1}],
        )

    def test_outer_scope_restored_after_inner_exits(self):
        # The merged annotation is cached and refreshed on scope enter/exit, so a node
        # created after an inner scope closes must get the outer annotation back rather than
        # the stale merge.
        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g, enable_annotations=True, annotation_config={"backend": "cupti"}
        ):
            with mark_kernels({"name": "outer"}):
                x = x + 1
                with mark_kernels({"name": "inner"}):
                    x = x + 1
                x = x + 1
        names = [
            entries[0]["name"] for _, entries in sorted(self._annotations().items())
        ]
        self.assertEqual(names, ["outer", "inner", "outer"])

    def test_conditional_body_work_warns_and_is_not_annotated(self):
        # Parity with the edge walk's conditional-body warning, but reported on the drop
        # rather than at scope entry: the handler cannot key body nodes to anything a trace
        # would match, so it counts them and disarm() says how many were lost. That covers
        # a scope *containing* a body as well as one inside it.
        from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

        x = torch.ones([2048], device="cuda")
        pred = torch.tensor(True, device="cuda")
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with self.assertWarnsRegex(UserWarning, "were not annotated"):
            with torch.cuda.graph(
                g, enable_annotations=True, annotation_config={"backend": "cupti"}
            ):
                with mark_kernels("region"):
                    z = x + 1
                    with _if_body(pred):
                        _ = torch.sqrt(z)
        g.instantiate()

        # Whatever was annotated is top-level: every key belongs to the exec graph, so none
        # of the body's nodes (which live in the body graph's id space) slipped in. The
        # exact top-level count is left alone -- _if_body emits kernels of its own.
        annotations = self._annotations()
        self.assertGreater(len(annotations), 0)
        exec_graph_id = g.get_graph_data()["exec_graph_id"]
        for tools_id in annotations:
            self.assertEqual(tools_id >> 32, exec_graph_id)

    def test_no_warning_without_body_work(self):
        # The counter must not leak across captures: a plain capture right after one that
        # dropped body nodes has to be silent.
        import warnings as _warnings

        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            with torch.cuda.graph(
                g, enable_annotations=True, annotation_config={"backend": "cupti"}
            ):
                with mark_kernels("region"):
                    x = x + 1

    def test_scopes_do_not_leak_across_captures(self):
        # A capture that raises mid-scope must not leave the scope open for the next one.
        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with torch.cuda.graph(
                g, enable_annotations=True, annotation_config={"backend": "cupti"}
            ):
                with mark_kernels("doomed"):
                    x = x + 1
                    raise RuntimeError("boom")

        _reset_kernel_annotations()
        g2 = torch.cuda.CUDAGraph()
        y = self._warm(torch.randn(64, 64, device="cuda"))
        with torch.cuda.graph(
            g2, enable_annotations=True, annotation_config={"backend": "cupti"}
        ):
            y = y + 1
        # No scope was open, so nothing should be attributed to the doomed one.
        self.assertEqual(self._annotations(), {})

    def test_failed_backward_does_not_leak_its_annotation(self):
        # A node that raises never runs its posthook, so the annotation its bracket
        # published stays on the module-global stack. If the failure is caught inside the
        # capture, every later unmarked node would inherit it.
        class Boom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                raise RuntimeError("boom")

        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        self._warm(x.detach())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g, enable_annotations=True, annotation_config={"backend": "cupti"}
        ):
            with mark_kernels("fwd"):
                y = Boom.apply(x)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                torch.autograd.grad(y.sum(), x)
            # Outside every scope, and after the failure: must be attributed to nothing.
            _ = x + 1

        recorded = [
            entry for entries in self._annotations().values() for entry in entries
        ]
        self.assertNotIn("autograd_phase", {k for entry in recorded for k in entry})
        # The forward scope's own kernels are still attributed.
        self.assertIn({"name": "fwd"}, recorded)

    def test_callback_disarmed_after_capture(self):
        from torch.cuda import _graph_node_callbacks

        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            g, enable_annotations=True, annotation_config={"backend": "cupti"}
        ):
            with mark_kernels("phase"):
                x = x + 1
        self.assertIsNone(_graph_node_callbacks._handler)
        before = len(self._annotations())
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        self.assertEqual(len(self._annotations()), before)

    def test_cupti_backend_requires_single_threaded_autograd(self):
        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        with torch.autograd.grad_mode.set_multithreading_enabled(True):
            with self.assertRaisesRegex(RuntimeError, "single-threaded autograd"):
                with torch.cuda.graph(
                    g, enable_annotations=True, annotation_config={"backend": "cupti"}
                ):
                    x = x + 1

    def test_auto_falls_back_when_multithreaded(self):
        # "auto" must never raise: multithreaded autograd just means the edge walk.
        from torch.cuda import _graph_annotations

        x = self._warm(torch.randn(64, 64, device="cuda"))
        g = torch.cuda.CUDAGraph()
        seen = []
        with torch.autograd.grad_mode.set_multithreading_enabled(True):
            with torch.cuda.graph(g, enable_annotations=True):
                seen.append(_graph_annotations._annotation_backend)
                with mark_kernels("phase"):
                    x = x + 1
        self.assertEqual(seen, ["edge_walk"])
        self.assertEqual(len(self._annotations()), 1)

    def test_invalid_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, r"annotation_config\['backend'\]"):
            torch.cuda.graph(
                torch.cuda.CUDAGraph(), annotation_config={"backend": "nope"}
            )
        # A misspelled or unsupported key must raise rather than silently leave the default
        # in place.
        with self.assertRaisesRegex(ValueError, "unrecognized annotation_config key"):
            torch.cuda.graph(
                torch.cuda.CUDAGraph(), annotation_config={"backend_name": "cupti"}
            )


if __name__ == "__main__":
    run_tests()
