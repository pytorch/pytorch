# Owner(s): ["oncall: profiler"]
"""Tests for ``NodeTimerObserver`` -- the minimal always-on per-graph-node timing
consumer of the CUPTI monitor. Collection runs through the monitor, so these need
CUDA + libcupti >= 13.3 (gated the same way as the rest of the monitor suite)."""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.utils._import_utils import _check_module_exists


TEST_CUDA = torch.cuda.is_available()
# cupti-python is pip-installable on ROCm hosts too, but CUPTI itself is a no-op
# there, so gate the monitor tests off ROCm as well.
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


def _cupti_version() -> int:
    if not TEST_CUPTI_PYTHON:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti

        return pylibcupti().get_version()
    except Exception:
        return 0


# The CUPTI monitor needs libcupti >= 13.3 (v2 user-defined records + populated
# ppRecordLayouts); a single >= 13.3 gate covers the whole monitor.
TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


def _capture_relu_graph() -> "torch.cuda.CUDAGraph":
    """Warm up, then capture a small in-place elementwise loop into a CUDA graph.
    In-place ops on a fixed buffer keep capture allocation-free; each :meth:`replay`
    re-runs the captured nodes, so their CUPTI graph_node_ids recur across replays."""
    x = torch.randn(512, 512, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            torch.relu_(x)
            x.mul_(0.5)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4):
            torch.relu_(x)
            x.mul_(0.5)
    return g


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiNodeTimerCUDA(TestCase):
    """NodeTimerObserver collection through the CUPTI monitor (not via profile)."""

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_collects_kernel_spans(self):
        # NodeTimerObserver is the minimal timing consumer: it registers just the
        # kernel timing fields (START/END/GRAPH_NODE_ID) and drain() returns flat
        # (graph_node_id, start, end) numpy columns. Eager kernels share node 0.
        import numpy as np

        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(256, 256, device="cuda")
            for _ in range(4):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            # drain()'s own flush is plain/best-effort, so deterministically deliver
            # everything via the monitor's sync flush, then drain.
            obs._monitor.flush(sync=True)
            gnode, start, end, stream = obs.drain()
            # drain() resets; with no new work the next drain is empty.
            _, start2, _, _ = obs.drain()
        finally:
            obs.close()

        self.assertGreater(len(start), 0)
        self.assertEqual(len(start), len(end))
        self.assertEqual(len(gnode), len(start))
        self.assertEqual(len(stream), len(start))
        self.assertEqual(len(start2), 0)
        # durations are non-negative, and the columns have the documented dtypes.
        self.assertTrue(bool((end >= start).all()))
        self.assertEqual(gnode.dtype, np.dtype("<u8"))
        self.assertEqual(start.dtype, np.dtype("<i8"))
        self.assertEqual(stream.dtype, np.dtype("<u8"))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_drain_annotated_eager(self):
        # With eager naming on (default), eager kernels bracketed by annotate(name) resolve
        # to that region via the correlation_id -> external_id -> name join, and
        # drain_annotated() returns {name: [(start, end), ...]}.
        from torch.profiler._cupti.observers.base import ObserverAnnotationSettings
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(
            annotations=ObserverAnnotationSettings(support_eager_annotations=True)
        )
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(128, 128, device="cuda")
            with obs.annotate("regionA"):
                for _ in range(3):
                    x = torch.relu(x @ x)
                x.sum().item()
                torch.cuda.synchronize()
            # drain_annotated()'s flush is best-effort; deliver deterministically first.
            obs._monitor.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertIn("regionA", spans)
        self.assertGreater(len(spans["regionA"]), 0)
        for start_ns, end_ns in spans["regionA"]:
            self.assertGreaterEqual(end_ns, start_ns)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_nested_resolves_to_enclosing_region(self):
        # A kernel launched under an inner (unnamed) external-correlation push -- as a
        # collective would push, leaf and innermost -- nested inside a named region
        # still resolves to that region. The single-kind record carries only the inner
        # id; node_timer recovers the enclosing region from the monitor's active-id
        # chain at dispatch.
        from torch.profiler._cupti.observers.base import ObserverAnnotationSettings
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(
            annotations=ObserverAnnotationSettings(support_eager_annotations=True)
        )
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        mon = obs._monitor
        try:
            x = torch.randn(128, 128, device="cuda")
            with obs.annotate("outer"):
                # Inner unnamed push (mimics a nested collective tag): innermost id,
                # not a named region -> the kernels must fall back to "outer".
                mon.push_external_correlation_id()
                try:
                    for _ in range(3):
                        x = torch.relu(x @ x)
                    x.sum().item()
                    torch.cuda.synchronize()
                finally:
                    mon.pop_external_correlation_id()
            mon.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertIn("outer", spans)
        self.assertGreater(len(spans["outer"]), 0)
        for start_ns, end_ns in spans["outer"]:
            self.assertGreaterEqual(end_ns, start_ns)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_drain_annotated_unnamed_bucket(self):
        # With no annotations configured, drain_annotated() doesn't drop or raise --
        # every span lands in the "" bucket.
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            x = torch.randn(128, 128, device="cuda")
            for _ in range(3):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            obs._monitor.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        self.assertEqual(set(spans), {""})
        self.assertGreater(len(spans[""]), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_collects_graph_node_spans(self):
        # CUDA-graph-captured kernels carry real (non-zero) graph_node_ids -- unlike
        # eager kernels (node 0) -- and the same node recurs across replays, so
        # consumers can key per-node timing. NodeTimerObserver must surface those ids.
        import numpy as np

        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            g = _capture_relu_graph()
            obs._monitor.flush(sync=True)
            obs.drain()  # discard warmup spans; keep only the replayed graph nodes
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            obs._monitor.flush(sync=True)
            gnode, start, end, stream = obs.drain()
        finally:
            obs.close()

        self.assertGreater(len(start), 0)
        if not bool((gnode != 0).any()):
            self.skipTest("driver did not populate CUPTI graph node ids")
        self.assertTrue(bool((end >= start).all()))
        # A captured node recurs once per replay, so some node id appears >= 2 times.
        _, counts = np.unique(gnode[gnode != 0], return_counts=True)
        self.assertTrue(bool((counts >= 2).any()))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_node_timer_drain_annotated_graph(self):
        # Graph-first naming: a graph_annotation_resolver maps each captured node id to
        # a region name, and drain_annotated() buckets the replayed spans under it --
        # no extra record kinds, and it survives replay (unlike the eager join).
        from torch.profiler._cupti.observers.base import ObserverAnnotationSettings
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(
            annotations=ObserverAnnotationSettings(
                graph_annotation_resolver=lambda nid: "graphregion" if nid else None
            )
        )
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            g = _capture_relu_graph()
            obs._monitor.flush(sync=True)
            obs.drain_annotated()  # discard warmup spans
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            obs._monitor.flush(sync=True)
            spans = obs.drain_annotated()
        finally:
            obs.close()

        if "graphregion" not in spans:
            self.skipTest("driver did not populate CUPTI graph node ids")
        self.assertGreater(len(spans["graphregion"]), 0)
        for start_ns, end_ns in spans["graphregion"]:
            self.assertGreaterEqual(end_ns, start_ns)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires >= 2 GPUs")
    def test_node_timer_collects_memcpy2_spans(self):
        # Peer-to-peer / cross-device copies surface under MEMCPY2 (not MEMCPY). CUPTI
        # only emits MEMCPY2 when MEMCPY is enabled too, so requesting MEMCPY2 implicitly
        # enables MEMCPY -- a cross-device copy_ then drains as a timed span.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver(kinds=(ActivityKind.MEMCPY2,))
        # MEMCPY2 pulls in MEMCPY implicitly (else CUPTI emits no MEMCPY2 records).
        self.assertIn(int(ActivityKind.MEMCPY), {int(k) for k in obs._kinds})
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")
        try:
            a = torch.randn(1024, 1024, device="cuda:0")
            b = torch.empty(1024, 1024, device="cuda:1")
            for _ in range(4):
                b.copy_(a)
            torch.cuda.synchronize()
            obs._monitor.flush(sync=True)
            gnode, start, end, stream = obs.drain()
        finally:
            obs.close()

        if len(start) == 0:
            self.skipTest("driver did not emit MEMCPY2 records for the P2P copies")
        self.assertEqual(len(start), len(end))
        self.assertEqual(len(gnode), len(start))
        self.assertEqual(len(stream), len(start))
        self.assertTrue(bool((end >= start).all()))


if __name__ == "__main__":
    run_tests()
