# Owner(s): ["oncall: profiler"]
"""Tests for the CUPTI activity monitor and its v2/user-defined-record codec.

Covers the monitor at every layer: the ``records`` field schema + ``decode`` codec
(pure, no CUDA), collection through ``CuptiMonitor`` directly (CUDA), the native
buffer-pool / v2-record-layout callbacks driven via ctypes, and the monitor *through*
``torch.profiler.profile`` (trace shape, op/kernel-name parity, record_shapes,
multithread, sync/async export, ...).
"""

import functools
import gzip
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import (
    kineto_available,
    profile,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.profiler._cupti.observers.observation_window import WindowFinalizerMixin
from torch.testing._internal.common_cuda import (
    SM100OrLater,
    TEST_CUDA,
    TEST_CUDA_GRAPH_TOOLS_ID,
    TEST_CUPTI as TEST_CUPTI_PYTHON,
    TEST_CUPTI_V13_3,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TemporaryFileName,
    TestCase,
)


def setUpModule():
    if (
        kineto_available()
        and torch.cuda.is_available()
        and ProfilerActivity.CUDA in supported_activities()
    ):
        # Kineto's process-global profiler cannot currently upgrade from a
        # CPU-only first initialization to CUDA-capable profiling. Prime it with
        # CUDA so CPU-only tests do not poison later CUDA profiler tests.
        x = torch.ones(1, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            x + x
            torch.cuda.synchronize()
        # Priming leaves libkineto holding the single process-wide CUPTI subscriber, so a
        # later cupti_monitor session can't subscribe (MULTIPLE_SUBSCRIBERS). Release it
        # via the documented cuptiFinalize hand-off -- no monitor exists yet, so this is
        # safe; libkineto re-subscribes on its next profile, so kineto tests are
        # unaffected. See pylibcupti().finalize.
        if TEST_CUPTI_V13_3:
            from torch.profiler._cupti.cupti_python import pylibcupti

            try:
                pylibcupti().finalize()
            except Exception:
                pass


def _isolated(test_fn):
    """Run a cupti_monitor test in a fresh subprocess. The monitor needs the single
    process-wide CUPTI subscriber, but libkineto grabs it for the process lifetime once
    any kineto-CUDA profile runs (setUpModule's prime, or another test in a full run),
    and decoded-record native state can leak between tests. Re-running the test alone in
    a child gives it a clean process: setUpModule's finalize releases the child's own
    prime, so the monitor subscribes, with no cross-test interference."""

    @functools.wraps(test_fn)
    def wrapper(self):
        if os.environ.get("PYTORCH_CUPTI_ISOLATED_CHILD") == "1":
            return test_fn(self)
        test_id = f"{type(self).__name__}.{test_fn.__name__}"
        env = {**os.environ, "PYTORCH_CUPTI_ISOLATED_CHILD": "1"}
        proc = subprocess.run(
            [sys.executable, __file__, test_id],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            self.fail(
                f"isolated subprocess for {test_id} failed (rc={proc.returncode})\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

    return wrapper


def _fresh_event_node_recorder(test):
    """The ``_EventNodeRecorder`` singleton, reset first (and again at teardown) so the test
    starts from an unarmed recorder with an empty map and leaves none behind."""
    from torch.profiler._cupti._event_nodes import _EventNodeRecorder, _reset_for_test

    _reset_for_test()
    test.addCleanup(_reset_for_test)
    return _EventNodeRecorder()


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiRecords(TestCase):
    """Pure monitor + metadata unit tests (no CUDA)."""

    def setUp(self):
        # CuptiMonitor is a process-wide singleton; drop it so each test builds a fresh one.
        from torch.profiler._cupti import monitor as _cupti_monitor

        _cupti_monitor._reset_for_test()
        self.addCleanup(_cupti_monitor._reset_for_test)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_monitor_normalize_activities(self):
        # A registration request resolves to (kinds, per-kind field selection): a
        # bare kind iterable means "all fields"; a field map selects fields, with
        # "all"/None expanding; *_FIELD_KIND (0) is always included.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import FIELD_REGISTRY, Kernel

        m = CuptiMonitor()
        kernel = ActivityKind.CONCURRENT_KERNEL
        memcpy = ActivityKind.MEMCPY
        all_kernel = frozenset(FIELD_REGISTRY[kernel]) | {0}
        all_memcpy = frozenset(FIELD_REGISTRY[memcpy]) | {0}

        kinds, fields = m._normalize_activities([kernel, memcpy])
        self.assertEqual(kinds, frozenset({kernel, memcpy}))
        self.assertEqual(fields[kernel], all_kernel)
        self.assertEqual(fields[memcpy], all_memcpy)

        kinds, fields = m._normalize_activities({kernel: {Kernel.START}, memcpy: "all"})
        self.assertEqual(fields[kernel], frozenset({0, int(Kernel.START)}))
        self.assertEqual(fields[memcpy], all_memcpy)

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires the cupti python bindings")
    def test_configure_and_get_config(self):
        # configure() sets the process-wide config get_config() reports; it is first-come-
        # first-serve. Pure config -- no session, only the cupti python bindings (not
        # libcupti). setUp reset the singleton, so this starts from the defaults: both
        # cadences off (-1, caller-driven) and the approx clock off.
        from torch.profiler._cupti import monitor as cupti_monitor

        cfg = cupti_monitor.get_config()
        self.assertEqual(cfg["buffer_size"], 4 * 1024 * 1024)
        self.assertEqual(cfg["background_flush_period_s"], -1.0)
        self.assertEqual(cfg["background_drain_period_s"], -1.0)
        self.assertFalse(cfg["use_approx_timestamps"])
        self.assertFalse(cfg["configured"])

        cupti_monitor.configure(
            buffer_size=2048, background_flush_period_s=0.5, use_approx_timestamps=True
        )
        cfg = cupti_monitor.get_config()
        self.assertEqual(cfg["buffer_size"], 2048)
        self.assertEqual(cfg["background_flush_period_s"], 0.5)
        self.assertEqual(
            cfg["background_drain_period_s"], -1.0
        )  # independent; still off
        self.assertTrue(cfg["use_approx_timestamps"])
        self.assertTrue(cfg["configured"])

        # First-come-first-serve: a second configure() is ignored.
        cupti_monitor.configure(buffer_size=4096)
        self.assertEqual(cupti_monitor.get_config()["buffer_size"], 2048)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_singleton_snapshots_config(self):
        # CuptiMonitor() is the process-wide singleton: it returns the one instance,
        # built once from the configure() settings.
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.monitor import CuptiMonitor

        cupti_monitor.configure(
            buffer_size=2048,
            background_flush_period_s=0.5,
            background_drain_period_s=2.0,
        )
        m = CuptiMonitor()
        self.assertEqual(m.buffer_size, 2048)
        self.assertEqual(m.background_flush_period_s, 0.5)
        self.assertEqual(m.background_drain_period_s, 2.0)
        self.assertIs(CuptiMonitor(), m)

        # Unconfigured, both cadences default off (caller-driven).
        cupti_monitor._reset_for_test()
        m = CuptiMonitor()
        self.assertEqual(m.background_flush_period_s, -1.0)
        self.assertEqual(m.background_drain_period_s, -1.0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_monitor_use_approx_timestamps_config(self):
        # use_approx_timestamps is a configure() setting the singleton snapshots: OFF by
        # default (the per-subscriber callback only times device records correctly when the
        # monitor is the first CUPTI consumer), opt-in on. Constructing loads libcupti (>= 13.3).
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.monitor import CuptiMonitor

        self.assertFalse(CuptiMonitor()._timestamp_callback_enabled)
        cupti_monitor._reset_for_test()
        cupti_monitor.configure(use_approx_timestamps=True)
        self.assertTrue(CuptiMonitor()._timestamp_callback_enabled)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_monitor_external_correlation_not_started(self):
        # External-correlation push/pop are no-ops until the monitor is started (no
        # subscriber yet), returning None rather than touching CUPTI's global stack.
        from torch.profiler._cupti.monitor import CuptiMonitor

        m = CuptiMonitor()
        self.assertFalse(m._started)
        self.assertIsNone(m.push_external_correlation_id())
        self.assertIsNone(m.pop_external_correlation_id())

    def test_external_correlation_id_mirror(self):
        # The native per-thread mirror of CUPTI's external-correlation stack lets
        # the current id be read (CUPTI has push/pop but no peek). Pure: the mirror
        # is host-side, no CUDA/CUPTI.
        m = torch._C._profiler._cupti_monitor
        while m.current_external_id():  # drain residue from a prior test (same thread)
            m.note_external_pop()
        self.assertEqual(m.current_external_id(), 0)
        m.note_external_push(7)
        self.assertEqual(m.current_external_id(), 7)
        m.note_external_push(9)
        self.assertEqual(m.current_external_id(), 9)  # top == innermost push
        self.assertEqual(m.note_external_pop(), 9)
        self.assertEqual(m.current_external_id(), 7)
        self.assertEqual(m.note_external_pop(), 7)
        self.assertEqual(m.current_external_id(), 0)
        self.assertEqual(m.note_external_pop(), 0)  # empty -> 0

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_external_id_chain_and_gc(self):
        # The push-time active-id chain: with one kind CUPTI tags a kernel with only
        # the innermost id, so the monitor snapshots the full active stack per id and
        # exposes external_id_chain(innermost) -> (outermost..innermost) for consumers
        # to recover enclosing contexts. Popped ids' chains are GC'd one generation
        # late (their records arrive after the pop). Drive the state directly -- no
        # CUDA/CUPTI.
        from torch.profiler._cupti.monitor import CuptiMonitor

        m = CuptiMonitor()
        # region id 1 encloses collective id 2 (as push_external_correlation_id would
        # record snapshotting this thread's live stack).
        m._id_chains[1] = (1,)
        m._id_chains[2] = (1, 2)
        # The collective's innermost id -> the full active set (region + collective).
        self.assertEqual(m.external_id_chain(2), (1, 2))
        self.assertEqual(m.external_id_chain(1), (1,))
        # An id with no snapshot falls back to itself (already GC'd, or not ours).
        self.assertEqual(m.external_id_chain(9), (9,))
        # GC keeps a popped id's chain through one drain, then drops it; a still-active
        # id (never popped) is retained.
        m._chains_gc_pending = [2]
        m._gc_external_chains()  # 2: pending -> ready, still present
        self.assertIn(2, m._id_chains)
        m._gc_external_chains()  # ready dropped
        self.assertNotIn(2, m._id_chains)
        self.assertIn(1, m._id_chains)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_drain_polls_pm_consumers(self):
        # Regression: PM-sampling frames must be pulled on the drain path. flush() and
        # the background drain loop both funnel through _drain_and_dispatch, so the PM
        # poll is folded there -- the native self-flush thread never touches the PM ring
        # (poll()/deliver are GIL work). Pure: a fake consumer, no CUDA/session.
        import numpy as np

        from torch.profiler._cupti.monitor import CuptiMonitor

        class _FakeHandle:
            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return {"start_ns": np.array([100, 200], dtype=np.int64)}

        m = CuptiMonitor()
        delivered: list = []
        handle = _FakeHandle()
        m._pm_consumers[lambda frame: delivered.append(frame)] = handle

        m._drain_and_dispatch()

        self.assertEqual(handle.polls, 1)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(list(delivered[0]["start_ns"]), [100, 200])

    def test_metadata_store_roundtrip(self):
        # The CollTrace-replacement metadata store: producers put a JSON object keyed
        # by the CURRENT external-correlation id on this thread's stack (pushed around
        # the op), so the put takes no id. Repeated puts under the same id MERGE. It
        # drains alongside the decoded records (folded into drain_decoded's return) so
        # the metadata and the records it annotates come from one snapshot. Pure
        # host-side, no CUDA/CUPTI.
        m = torch._C._profiler._cupti_monitor
        m.drain_decoded()  # clear any residue
        # No id on the stack -> put is a no-op.
        m.metadata_put_external(json.dumps({"func": "AllReduce"}))
        self.assertEqual(m.drain_decoded(), ([], {}))
        # Pushed id keys the object; puts merge recursively (mirrors several
        # producers each contributing fields, incl. nested objects, for one op).
        m.note_external_push(42)
        m.metadata_put_external(json.dumps({"func": "AllReduce", "nested": {"a": 1}}))
        m.metadata_put_external(json.dumps({"count": 4096, "nested": {"b": 2}}))
        m.note_external_pop()
        groups, ext_meta = m.drain_decoded()
        self.assertEqual(groups, [])  # no decode activity
        self.assertEqual(set(ext_meta), {42})
        # Top-level union + nested objects combined (recursive merge).
        self.assertEqual(
            json.loads(ext_meta[42]),
            {"func": "AllReduce", "count": 4096, "nested": {"a": 1, "b": 2}},
        )
        # drained -> empty
        self.assertEqual(m.drain_decoded(), ([], {}))

    def test_attach_metadata_join(self):
        # The CollTrace-replacement join: a blob keyed by external_id is attached as a
        # "metadata" column onto the GPU-op kinds whose correlation_id maps (via the
        # window's EXTERNAL_CORRELATION columns -- all ours, the monitor is the sole
        # pusher) to that external_id. Pure host-side, columnar.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_metadata

        columns = {
            "kernel": {
                "correlation_id": np.array([100, 200, 300]),  # 100->ext 42->blob
                "graph_node_id": np.array([0, 0, 0]),
            },
            "external_correlation": {
                "correlation_id": np.array([100, 200]),
                "external_id": np.array([42, 43]),
            },
        }
        blob = json.dumps({"func": "AllReduce", "count": 4096})
        _attach_metadata(columns, {42: blob}, None)
        meta = columns["kernel"]["metadata"]
        self.assertEqual(json.loads(meta[0]), {"func": "AllReduce", "count": 4096})
        self.assertIsNone(meta[1])  # external id has no blob
        self.assertIsNone(meta[2])  # no correlation -> external mapping
        # No metadata + no resolver is a no-op (the non-collective path pays nothing).
        clean = {
            "kernel": {
                "correlation_id": np.array([100]),
                "graph_node_id": np.array([0]),
            },
            "external_correlation": {
                "correlation_id": np.array([100]),
                "external_id": np.array([42]),
            },
        }
        _attach_metadata(clean, {}, None)
        self.assertNotIn("metadata", clean["kernel"])

    def test_attach_metadata_graph_resolver(self):
        # CUDA-graph-captured collectives have no replay-time external correlation;
        # their blob is resolved by graph_node_id via the metadata resolver (the same
        # mechanism as graph annotation names). Pure host-side, columnar.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_metadata

        blob = json.dumps({"func": "AllReduce"})
        registry = {7: blob}  # stack-managed graph_node_id -> blob
        resolver = registry.get

        # Replay op: stable node 7, fresh correlation id, no EXTERNAL_CORRELATION
        # -> the eager join misses but the resolver hits.
        replay = {
            "kernel": {
                "correlation_id": np.array([999]),
                "graph_node_id": np.array([7]),
            }
        }
        _attach_metadata(replay, {}, resolver)
        self.assertEqual(
            json.loads(replay["kernel"]["metadata"][0]), {"func": "AllReduce"}
        )

        # Unknown node -> no blob (None).
        other = {
            "kernel": {
                "correlation_id": np.array([1000]),
                "graph_node_id": np.array([8]),
            }
        }
        _attach_metadata(other, {}, resolver)
        self.assertIsNone(other["kernel"]["metadata"][0])

        # Eager wins over the resolver when both could resolve (and avoids calling it).
        both = {
            "kernel": {
                "correlation_id": np.array([100]),
                "graph_node_id": np.array([7]),
            },
            "external_correlation": {
                "correlation_id": np.array([100]),
                "external_id": np.array([42]),
            },
        }
        eager_blob = json.dumps({"func": "ReduceScatter"})
        _attach_metadata(both, {42: eager_blob}, resolver)
        self.assertEqual(
            json.loads(both["kernel"]["metadata"][0]), {"func": "ReduceScatter"}
        )

    def test_recorder_keeps_node_id_order_and_refuses_opaque_bodies(self):
        # The recorder takes event nodes in the order get_graph_data() returns them (node-id
        # == node-creation order) with no DAG-derived sorting: sync id follows node id, not
        # execution order, so ordering by dependency depth mislabels nodes on any fork/join
        # graph. Graphs with child/conditional bodies are refused outright -- get_graph_data()
        # does not descend into them, so their event nodes are invisible here. Pure host-side.
        def node(ntype, tid, deps):
            return {"node_type": ntype, "tools_id": tid, "dependencies": deps}

        class FakeGraph:
            def __init__(self, nodes):
                self._nodes = nodes
                self._recorded_exec_ids: set[int] = set()

            def get_graph_data(self):
                return {"nodes": self._nodes, "exec_graph_id": 7}

        # Fork/join: the deeper branch's event node is created FIRST (lower node id) but sits
        # at greater dependency depth. Node-id order must win.
        deep_first = [
            node("kernel", (7 << 32) | 0, []),
            node("kernel", (7 << 32) | 1, [0]),
            node("kernel", (7 << 32) | 2, [1]),
            node("event_record", (7 << 32) | 3, [2]),  # deep branch, created first
            node("event_record", (7 << 32) | 4, [0]),  # shallow branch, created second
        ]
        r = _fresh_event_node_recorder(self)
        r._on_instantiate(FakeGraph(deep_first))
        self.assertEqual(
            r.graph_event_nodes[7], [(7 << 32) | 3, (7 << 32) | 4]
        )  # NOT depth order, which would invert these

        # No event nodes -> nothing tracked at all.
        r2 = _fresh_event_node_recorder(self)
        r2._on_instantiate(FakeGraph([node("kernel", (7 << 32) | 0, [])]))
        self.assertNotIn(7, r2.graph_event_nodes)

        # Opaque bodies -> refuse, even though a top-level event node is present.
        for opaque in ("child_graph", "conditional"):
            r3 = _fresh_event_node_recorder(self)
            r3._on_instantiate(
                FakeGraph(
                    [
                        node("event_record", (7 << 32) | 0, []),
                        node(opaque, (7 << 32) | 1, [0]),
                    ]
                )
            )
            self.assertNotIn(7, r3.graph_event_nodes, opaque)

    def test_event_node_recorder_purge(self):
        # The recorder holds each graph's ordered event nodes; purge drops a destroyed graph.
        r = _fresh_event_node_recorder(self)
        r.graph_event_nodes = {7: [11, 13], 8: None}
        r.purge_exec_ids({7})
        self.assertNotIn(7, r.graph_event_nodes)
        self.assertIn(8, r.graph_event_nodes)

    def test_graph_recorders_are_singletons(self):
        # Both instantiate-hook recorders are process-wide singletons (like CuptiMonitor): a
        # later construction hands back the one instance WITHOUT re-initializing it, which is
        # what lets an observer created mid-run (at prepare_trace) share the map a recorder
        # armed before warm-up capture has been filling.
        from torch.profiler._cupti._event_nodes import _EventNodeRecorder
        from torch.profiler._cupti._graph_deps import (
            _GraphDependencyRecorder,
            _reset_for_test,
        )

        r = _fresh_event_node_recorder(self)
        r.graph_event_nodes[7] = [11]
        self.assertIs(_EventNodeRecorder(), r)
        self.assertEqual(_EventNodeRecorder().graph_event_nodes[7], [11])

        _reset_for_test()
        self.addCleanup(_reset_for_test)
        d = _GraphDependencyRecorder()
        d.deps[9] = [8]
        self.assertIs(_GraphDependencyRecorder(), d)
        self.assertEqual(_GraphDependencyRecorder().deps[9], [8])

    def test_resolve_window(self):
        # The window orchestration: keys each launch by correlation id -> exec graph id (from a
        # graphed work record's graph_node_id), orders that launch's event records by sync id,
        # and resolves each record to its node POSITIONALLY. Pure (no numpy/CUDA).
        from torch.profiler._cupti._event_nodes import resolve_window

        exec_id = 7
        n0, n1 = (exec_id << 32) | 11, (exec_id << 32) | 13
        r = _fresh_event_node_recorder(self)
        r.graph_event_nodes = {exec_id: [n0, n1]}

        corr_exec_pairs = [
            (900, (exec_id << 32) | 55)
        ]  # a graphed kernel in launch 900
        # Two event records for launch 900, delivered out of sync-id order.
        event_rows = [(900, 2), (900, 1)]
        resolved = resolve_window(r, corr_exec_pairs, event_rows)
        self.assertEqual(resolved, [n1, n0])  # sync 2 -> node1, sync 1 -> node0

        # A recycled CUDA event object would give both records the same event_id; positional
        # resolution keys on sync-id order, not the object, so the two records still map to
        # distinct nodes instead of collapsing onto the first.
        second = resolve_window(r, corr_exec_pairs, [(900, 10), (900, 20)])
        self.assertEqual(second, [n0, n1])

        # A launch whose event-record count does not match the graph's stays unresolved.
        self.assertEqual(resolve_window(r, corr_exec_pairs, [(900, 1)]), [None])

        # Event records with no matching graphed work record (unknown launch) stay unresolved.
        self.assertEqual(resolve_window(r, [], [(0, 0)]), [None])

    def test_add_graph_event_node_spans(self):
        # The window-bucketed event-node span frame keeps only rows resolved to a graph node
        # (graph_node_id != 0) whose device timestamp is in [start, boundary), and builds point
        # spans with graph_id = node_id >> 32. Pure host-side.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _add_graph_event_node_spans

        node = (7 << 32) | 11
        cols = {
            "cuda_event": {
                "event_id": np.array([5, 6, 7], dtype=np.int64),
                # row 1 resolved+in-window; row 2 unresolved; row 3 resolved but out of window
                "graph_node_id": np.array([node, 0, node], dtype=np.int64),
                "start_ns": np.array([1500, 1500, 9999], dtype=np.int64),
                "end_ns": np.array([1500, 1500, 9999], dtype=np.int64),
                "device_id": np.array([0, 0, 0], dtype=np.int64),
                "context_id": np.array([1, 1, 1], dtype=np.int64),
                "stream_id": np.array([7, 7, 7], dtype=np.int64),
                "correlation_id": np.array([900, 900, 901], dtype=np.int64),
                "annotation": np.array([None, None, None], dtype=object),
            }
        }
        _add_graph_event_node_spans(cols, start_ns=1000, boundary_ns=2000)
        gen = cols["graph_event_node"]
        self.assertEqual(
            gen["graph_node_id"].tolist(), [node]
        )  # only the first row survives
        self.assertEqual(gen["graph_id"].tolist(), [7])
        self.assertEqual(gen["start_ns"].tolist(), [1500])
        self.assertEqual(gen["stream_id"].tolist(), [7])

        # No resolved rows -> no frame added at all.
        cols2 = {
            "cuda_event": {
                "graph_node_id": np.array([0], dtype=np.int64),
                "start_ns": np.array([1500], dtype=np.int64),
                "end_ns": np.array([1500], dtype=np.int64),
                "device_id": np.array([0], dtype=np.int64),
                "context_id": np.array([0], dtype=np.int64),
                "stream_id": np.array([0], dtype=np.int64),
                "correlation_id": np.array([0], dtype=np.int64),
                "annotation": np.array([None], dtype=object),
            }
        }
        _add_graph_event_node_spans(cols2, 1000, 2000)
        self.assertNotIn("graph_event_node", cols2)

    def test_add_graph_event_node_spans_attaches_lane_columns(self):
        # The span frame is synthesized after the _COLUMN_BUILDERS lane pass has run, so it must
        # attach its own (logical_lane, lane_name); without them the export's reassignment is
        # gated off and the event node strands on its capture stream while the collective's
        # kernel moves to the process-group lane.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _add_graph_event_node_spans

        node = (7 << 32) | 11

        def cols_for():
            return {
                "cuda_event": {
                    "event_id": np.array([5], dtype=np.int64),
                    "graph_node_id": np.array([node], dtype=np.int64),
                    "start_ns": np.array([1500], dtype=np.int64),
                    "end_ns": np.array([1500], dtype=np.int64),
                    "device_id": np.array([0], dtype=np.int64),
                    "context_id": np.array([1], dtype=np.int64),
                    "stream_id": np.array([377], dtype=np.int64),
                    "correlation_id": np.array([900], dtype=np.int64),
                    "annotation": np.array([None], dtype=object),
                }
            }

        # A resolver moves the node off its capture stream onto the named logical lane.
        cols = cols_for()
        _add_graph_event_node_spans(
            cols, 1000, 2000, lane_resolver=lambda g: (61, "DP")
        )
        gen = cols["graph_event_node"]
        self.assertEqual(gen["logical_lane"].tolist(), [61])
        self.assertEqual(gen["lane_name"].tolist(), ["DP"])
        self.assertEqual(gen["stream_id"].tolist(), [377])  # capture stream preserved

        # No resolver -> no lane columns, and the export leaves the row on its CUDA stream.
        plain = cols_for()
        _add_graph_event_node_spans(plain, 1000, 2000)
        self.assertNotIn("logical_lane", plain["graph_event_node"])

    def test_attach_event_node_ids_uses_installed_resolver(self):
        # The annotation column must come from the installed resolver (the same one the GPU-op
        # column builders use), not a direct read of torch.cuda._graph_annotations -- an
        # installed resolver may merge a node's annotation list into one dict, and only a dict
        # gets spread into the exported event's args.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_event_node_ids

        exec_id = 7
        n0 = (exec_id << 32) | 11
        r = _fresh_event_node_recorder(self)
        r.graph_event_nodes = {exec_id: [n0]}

        columns = {
            "kernel": {
                "correlation_id": np.array([900], dtype=np.int64),
                "graph_node_id": np.array([(exec_id << 32) | 55], dtype=np.int64),
            },
            "cuda_event": {
                "event_id": np.array([5], dtype=np.int64),
                "correlation_id": np.array([900], dtype=np.int64),
                "cuda_event_sync_id": np.array([1], dtype=np.int64),
            },
        }
        seen: list[int] = []

        def resolver(g):
            seen.append(g)
            return {"roofline_id": "fsdp.all_gather_collective", "stream": 61}

        _attach_event_node_ids(columns, r, resolver)
        ce = columns["cuda_event"]
        self.assertEqual(ce["graph_node_id"].tolist(), [n0])
        self.assertEqual(seen, [n0])
        self.assertEqual(
            ce["annotation"].tolist(),
            [{"roofline_id": "fsdp.all_gather_collective", "stream": 61}],
        )

    def test_graph_event_node_rendered_and_arrowed(self):
        # A resolved graph event-record node renders as an "EventRecord" point span under cat
        # "graph_event_node" on its stream lane, and (with graph_deps) is a dependency-arrow
        # endpoint: an arrow flows from the producing kernel to the event node. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _FLOW_CATEGORY,
            _trace_window_entries,
        )

        def i64(*v):
            return np.array(v, dtype=np.int64)

        KA = (1 << 32) | 101  # producing kernel node
        EN = (1 << 32) | 102  # event-record node depending on it
        columns = {
            "kernel": {
                "start_ns": i64(1000),
                "end_ns": i64(2000),
                "device_id": i64(0),
                "context_id": i64(1),
                "stream_id": i64(7),
                "correlation_id": i64(11),
                "graph_id": i64(1),
                "graph_node_id": i64(KA),
                "name": np.array(["producer"], dtype=object),
                "annotation": np.array([None], dtype=object),
                "grid_x": i64(1),
                "grid_y": i64(1),
                "grid_z": i64(1),
                "block_x": i64(1),
                "block_y": i64(1),
                "block_z": i64(1),
                "registers_per_thread": i64(0),
                "static_shared_memory": i64(0),
                "dynamic_shared_memory": i64(0),
                "priority": i64(0),
                "queued": i64(0),
                "channel": i64(0),
                "channel_type": i64(0),
            },
            "graph_event_node": {
                "start_ns": i64(2500),
                "end_ns": i64(2500),
                "device_id": i64(0),
                "context_id": i64(1),
                "stream_id": i64(7),
                "correlation_id": i64(11),
                "graph_id": i64(1),
                "graph_node_id": i64(EN),
                "annotation": np.array([None], dtype=object),
            },
        }
        # The event-record node records cudaEvent 0xABC (from graph topology); the EventRecord
        # span is tagged with it so, together with the arrow, the awaited event is visible.
        window = {
            "columns": columns,
            "graph_deps": {EN: [KA]},
            "graph_event_record_events": {EN: 0xABC},
        }
        _, events = _trace_window_entries(window, base_ns=0)
        en = [e for e in events if e.get("cat") == "graph_event_node"]
        self.assertEqual(len(en), 1)
        self.assertEqual(en[0]["name"], "EventRecord")
        self.assertEqual(en[0]["pid"], 0)  # device lane
        self.assertEqual(en[0]["args"]["stream"], 7)
        self.assertEqual(en[0]["args"]["graph node id"], 102)
        self.assertEqual(en[0]["args"]["cuda event"], hex(0xABC))
        # A dependency arrow terminates ON the event node: some flow-finish endpoint lands at
        # the event node's start (kernel -> event_node). _FLOW_CATEGORY is shared by the
        # CPU-launch and graph-dep flows, so match on the event node's timestamp.
        finish_ts = {
            e["ts"]
            for e in events
            if e.get("cat") == _FLOW_CATEGORY and e.get("ph") == "f"
        }
        self.assertIn(2500 / 1000.0, finish_ts)

    def test_attach_event_node_ids(self):
        # End-to-end host-side: given a window's kernel + cuda_event columns, learn the
        # mapping (records ordered by sync id, launch keyed by correlation id -> exec graph id
        # from the kernel's graph_node_id) and attach graph_node_id + annotation to the
        # cuda_event frame.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import _attach_event_node_ids

        exec_id = 7
        n0, n1 = (exec_id << 32) | 11, (exec_id << 32) | 13
        r = _fresh_event_node_recorder(self)
        r.graph_event_nodes = {exec_id: [n0, n1]}

        columns = {
            "kernel": {
                "correlation_id": np.array([900], dtype=np.int64),
                "graph_node_id": np.array([(exec_id << 32) | 55], dtype=np.int64),
            },
            # Records delivered out of execution order: sorting by sync id recovers it.
            "cuda_event": {
                "event_id": np.array([6, 5], dtype=np.int64),
                "correlation_id": np.array([900, 900], dtype=np.int64),
                "cuda_event_sync_id": np.array([2, 1], dtype=np.int64),
            },
        }
        _attach_event_node_ids(columns, r)
        ce = columns["cuda_event"]
        # Rows arrive out of execution order; resolution is POSITIONAL by sync id, so the
        # sync-1 row (event_id 5) takes the first ordered node n0 and the sync-2 row
        # (event_id 6) takes n1 -- realigned to the input order [6, 5] -> [n1, n0].
        # There is deliberately no event_id -> node lookup to assert against: a producer
        # recycles one cudaEvent_t across nodes, so event_id is stable but not unique to a
        # node, and keying on it would collapse every recycled record onto one node.
        self.assertEqual(ce["graph_node_id"].tolist(), [n1, n0])
        self.assertEqual(ce["annotation"].tolist(), [None, None])

        # No recorder -> no-op (bridge disabled).
        plain = {"cuda_event": {"event_id": np.array([1], dtype=np.int64)}}
        _attach_event_node_ids(plain, None)
        self.assertNotIn("graph_node_id", plain["cuda_event"])

    def test_window_graph_deps_collapses_event_nodes(self):
        # A dependency arrow must span a kernel -> event_record -> wait_event -> kernel chain
        # (event/wait nodes never render as spans, so they can't be arrow endpoints). The window
        # dep resolver collapses those non-present nodes to the nearest rendered ancestor.
        import numpy as np

        from torch.profiler._cupti.observers.profiler import (
            _nearest_present_preds,
            _window_graph_deps,
        )

        # kernelA -> event_record -> wait_event -> kernelB
        KA, E, W, KB = 10, 11, 12, 13
        # node -> predecessors, as the _graph_deps recorder stores them
        raw = {E: [KA], W: [E], KB: [W]}
        resolver = raw.get

        # Direct: KB's predecessor collapses through W and E to the rendered kernel KA.
        self.assertEqual(_nearest_present_preds(resolver, KB, {KA, KB}), [KA])
        # A root kernel has no present predecessors.
        self.assertEqual(_nearest_present_preds(resolver, KA, {KA, KB}), [])

        # Through the window API: only kernels KA/KB are present (rendered); the arrow survives.
        columns = {"kernel": {"graph_node_id": np.array([KA, KB], dtype=np.int64)}}
        self.assertEqual(_window_graph_deps(resolver, columns), {KB: [KA]})

        # Backward compat: a plain kernel->kernel edge (no event nodes) is unchanged.
        self.assertEqual(
            _window_graph_deps(
                {KB: [KA]}.get,
                {"kernel": {"graph_node_id": np.array([KA, KB], dtype=np.int64)}},
            ),
            {KB: [KA]},
        )

        # Fan-in: KB depends on two chains, each through an event node, to two kernels.
        KA2 = 20
        raw2 = {E: [KA], 21: [KA2], KB: [E, 21]}  # 21 is a second event_record
        self.assertEqual(
            sorted(_nearest_present_preds(raw2.get, KB, {KA, KA2, KB})), [KA, KA2]
        )

    def test_metadata_blob_rendered_into_trace_args(self):
        # The comms descriptor blob attached as the "metadata" column is spread into the
        # chrome-trace kernel args (so func/count/... show up), the same way a dict
        # annotation is. Drives the columnar merge directly (no CUDA).
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _trace_window_entries

        def col(v):
            return np.array([v], dtype=np.int64)

        blob = json.dumps({"func": "AllReduce", "count": 4096})
        columns = {
            "kernel": {
                "start_ns": col(1000),
                "end_ns": col(2000),
                "device_id": col(0),
                "context_id": col(1),
                "stream_id": col(7),
                "correlation_id": col(5),
                "graph_id": col(0),
                "graph_node_id": col(0),
                "name": np.array(["ncclDevKernel"], dtype=object),
                "annotation": np.array([None], dtype=object),
                "grid_x": col(1),
                "grid_y": col(1),
                "grid_z": col(1),
                "block_x": col(1),
                "block_y": col(1),
                "block_z": col(1),
                "registers_per_thread": col(0),
                "static_shared_memory": col(0),
                "dynamic_shared_memory": col(0),
                "priority": col(0),
                "queued": col(0),
                "channel": col(0),
                "channel_type": col(0),
                "metadata": np.array([blob], dtype=object),
            }
        }
        _, events = _trace_window_entries({"columns": columns}, base_ns=0)
        kernels = [e for e in events if e.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 1)
        args = kernels[0]["args"]
        self.assertEqual(args["func"], "AllReduce")
        self.assertEqual(args["count"], 4096)

    def test_graph_host_node_rendered_in_trace(self):
        # A CUPTI GRAPH_HOST_NODE record (a CPU callback run as a CUDA-graph node) renders as a
        # "HostNode" span under cat "graph_host_node" on the waiting stream's lane, with its
        # graph id / node id and dict annotation patched on like the other graphed ops. When a
        # host_fn name/address is resolved (recorded at graph instantiate), the span is named
        # "HostNode: <fn>" and both surface in args. Drives the columnar merge directly (no CUDA).
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _trace_window_entries

        def col(v):
            return np.array([v], dtype=np.int64)

        columns = {
            "graph_host_node": {
                "start_ns": col(1000),
                "end_ns": col(2000),
                "device_id": col(0),
                "context_id": col(1),
                "stream_id": col(7),
                "correlation_id": col(9),
                "graph_id": col(1),
                # graph id packed into the upper 32 bits; only the lower node id is surfaced
                "graph_node_id": col((1 << 32) | 202),
                "annotation": np.array([{"ann_id": "host"}], dtype=object),
                "process_id": col(4321),
                "thread_id": col(8765),
                "host_fn": np.array(["free"], dtype=object),
                "host_fn_addr": col(0x7FABCD),
            }
        }
        _, events = _trace_window_entries({"columns": columns}, base_ns=0)
        host = [e for e in events if e.get("cat") == "graph_host_node"]
        self.assertEqual(len(host), 1)
        ev = host[0]
        self.assertEqual(ev["name"], "HostNode: free")
        self.assertEqual(ev["pid"], 0)  # rendered on the device lane
        args = ev["args"]
        self.assertEqual(args["stream"], 7)
        self.assertEqual(args["host process"], 4321)
        self.assertEqual(args["host thread"], 8765)
        self.assertEqual(args["graph id"], 1)
        self.assertEqual(args["graph node id"], 202)
        self.assertEqual(args["ann_id"], "host")  # dict annotation spread into args
        self.assertEqual(args["host fn"], "free")
        self.assertEqual(args["host fn addr"], hex(0x7FABCD))

    def test_graph_kernel_reassigned_to_logical_lane(self):
        # A graphed kernel the lane resolver maps to a logical lane is placed on that lane
        # (tid + args["stream"]), its real CUDA stream is preserved as original_stream, the
        # lane is named by the resolver, and its ac2g flow arrow follows -- all in the single
        # merge pass, so no read-trace/reassign/rewrite round trip is needed. A kernel the
        # resolver leaves on its CUDA stream is untouched (no original_stream) and its lane
        # keeps the default "stream N" name. Reassignment is driven by the logical_lane /
        # lane_name columns the observer fills from the lane resolver (supplied here
        # directly). No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _trace_window_entries

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        columns = {
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "context_id": i64(1, 1),
                "stream_id": i64(7, 7),  # both replayed on the capture stream
                "correlation_id": i64(11, 12),
                "graph_id": i64(1, 1),
                # graph id packed into the upper 32 bits; only the lower node id is kept
                "graph_node_id": i64((1 << 32) | 101, (1 << 32) | 102),
                "name": np.array(["graphKernelA", "graphKernelB"], dtype=object),
                "annotation": np.array(
                    [{"ann_id": "a"}, {"ann_id": "b"}], dtype=object
                ),
                # resolver output: A -> lane 8 named "side comms"; B -> its own stream 7 (no move)
                "logical_lane": i64(8, 7),
                "lane_name": np.array(["side comms", None], dtype=object),
                "grid_x": i64(1, 1),
                "grid_y": i64(1, 1),
                "grid_z": i64(1, 1),
                "block_x": i64(1, 1),
                "block_y": i64(1, 1),
                "block_z": i64(1, 1),
                "registers_per_thread": i64(0, 0),
                "static_shared_memory": i64(0, 0),
                "dynamic_shared_memory": i64(0, 0),
                "priority": i64(0, 0),
                "queued": i64(0, 0),
                "channel": i64(0, 0),
                "channel_type": i64(0, 0),
            }
        }
        metadata, events = _trace_window_entries({"columns": columns}, base_ns=0)
        kernels = {e["name"]: e for e in events if e.get("cat") == "kernel"}

        # resolved to a logical lane: moved onto lane 8, real CUDA stream preserved
        a = kernels["graphKernelA"]
        self.assertEqual(a["tid"], 8)
        self.assertEqual(a["args"]["stream"], 8)
        self.assertEqual(a["args"]["original_stream"], 7)
        self.assertEqual(a["args"]["ann_id"], "a")
        self.assertEqual(a["args"]["graph node id"], 101)

        # resolver left it on its CUDA stream lane: untouched, no original_stream
        b = kernels["graphKernelB"]
        self.assertEqual(b["tid"], 7)
        self.assertEqual(b["args"]["stream"], 7)
        self.assertNotIn("original_stream", b["args"])

        # the ac2g flow arrow for the reassigned kernel follows it to lane 8
        flow_by_id = {
            e["id"]: e for e in events if e.get("cat") == "ac2g" and e.get("ph") == "f"
        }
        self.assertEqual(flow_by_id[11]["tid"], 8)
        self.assertEqual(flow_by_id[12]["tid"], 7)

        # the reassigned lane is named by the resolver; the default lane keeps "stream N"
        names = {
            (e["pid"], e["tid"]): e["args"]["name"]
            for e in metadata
            if e.get("ph") == "M" and e.get("name") == "thread_name"
        }
        self.assertEqual(names[(0, 8)], "side comms")
        self.assertEqual(names[(0, 7)].strip(), "stream 7")

    def test_gpu_user_annotation_stays_on_capture_stream(self):
        # A GPU-side user annotation stays on its kernels' real capture stream, never a lane the
        # resolver reassigned kernels onto. But when a capture stream has no work left to
        # show (all its ops moved to a logical lane), the span would be orphaned on an empty
        # lane, so it is dropped instead. A stream that still has non-reassigned work keeps
        # its annotation. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _gpu_user_annotation_events

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        columns = {
            "external_correlation": {
                "correlation_id": i64(11, 12, 13, 14),
                "user_external_id": i64(555, 666, 777, 999),
            },
            "kernel": {
                "correlation_id": i64(11, 12, 13, 14),
                "device_id": i64(0, 0, 0, 0),
                "stream_id": i64(7, 9, 5, 5),
                "start_ns": i64(1000, 3000, 5000, 5500),
                "end_ns": i64(2000, 4000, 6000, 6500),
                # k11 graphed -> lane 8 (orphans stream 7); k12 eager on 9; k13 graphed -> lane
                # 6 but k14 is eager on the same stream 5, so stream 5 still shows work.
                "graph_node_id": i64(101, 0, 103, 0),
                "logical_lane": i64(8, 9, 6, 5),
            },
        }
        trace_window = {
            "columns": columns,
            "user_annotations": {
                555: "all_reduce",
                666: "matmul",
                777: "reduce_scatter",
            },
        }
        events = _gpu_user_annotation_events(trace_window, base_ns=0)
        by_name = {e["name"]: e for e in events}

        # graphed op moved off stream 7 and nothing else renders there -> annotation dropped
        self.assertNotIn("all_reduce", by_name)
        # eager: stays on the kernel's capture stream
        self.assertEqual(by_name["matmul"]["cat"], "gpu_user_annotation")
        self.assertEqual(by_name["matmul"]["tid"], 9)
        # stream 5 still has non-reassigned work (k14), so its annotation is kept there
        self.assertEqual(by_name["reduce_scatter"]["tid"], 5)

    def test_graph_dependency_flows_json(self):
        # CUDA-graph node->node dependency arrows in the JSON export: one flow per edge from the
        # predecessor's end to the successor's start, resolved within the same replay
        # (correlation_id) so arrows never cross replays, on the ops' display lanes. No CUDA.
        from torch.profiler._cupti.monitor_trace import _graph_dependency_flow_events

        # replay 100: node 11 on lane 7 [1000,2000]; node 12 on lane 8 [3000,4000], 12 -> 11.
        # replay 200 repeats the same nodes and stays self-contained.
        node_rows = [
            (100, 11, 0, 7, 1000, 2000),
            (100, 12, 0, 8, 3000, 4000),
            (200, 11, 0, 7, 5000, 6000),
            (200, 12, 0, 8, 7000, 8000),
        ]
        events = _graph_dependency_flow_events(node_rows, {12: [11]}, base_ns=0)
        starts = [e for e in events if e["ph"] == "s"]
        fins = [e for e in events if e["ph"] == "f"]
        self.assertEqual(len(starts), 2)  # one edge per replay
        self.assertEqual(len(fins), 2)
        s0 = starts[0]
        f0 = next(f for f in fins if f["id"] == s0["id"])
        # arrow leaves the predecessor's end on its lane and lands on the successor's start
        self.assertEqual((s0["tid"], s0["ts"]), (7, 2.0))
        self.assertEqual((f0["tid"], f0["ts"], f0["bp"]), (8, 3.0, "e"))
        self.assertNotEqual(
            starts[0]["id"], starts[1]["id"]
        )  # replays don't share a flow
        # no recorded dependencies -> no flows
        self.assertEqual(_graph_dependency_flow_events(node_rows, {}, base_ns=0), [])

        # Clock skew: predecessor 13 ends at 4400 -- AFTER successor 14 starts at 3500 -- so the
        # flow goes end(4.4) -> start(3.5), i.e. backwards. The JSON export renders that oddly (a
        # known limitation, see the function docstring); the .pftrace export does not.
        skewed = [
            (300, 13, 0, 7, 1000, 4400),
            (300, 14, 0, 8, 3500, 4500),
        ]
        ev = _graph_dependency_flow_events(skewed, {14: [13]}, base_ns=0)
        s = next(e for e in ev if e["ph"] == "s")
        f = next(e for e in ev if e["ph"] == "f")
        self.assertEqual(
            (s["ts"], f["ts"]), (4.4, 3.5)
        )  # end -> start, backwards under skew

    def test_cpu_launch_flow_targets_only_graph_roots(self):
        # With graph node->node arrows drawn (graph_deps present), the CPU-launch -> GPU-op flow
        # links only to root graph nodes; a non-root node gets its incoming edge from the dep
        # arrow, so the cudaGraphLaunch does not fan out to every replayed kernel. Diamond, one
        # replay (shared correlation): A (root) -> B, A -> C, {B, C} -> D. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _trace_window_entries

        def i64(*v):
            return np.array(v, dtype=np.int64)

        A, B, C, D, corr = 1001, 1002, 1003, 1004, 77
        kernel = {
            "start_ns": i64(1000, 3000, 3000, 5000),  # A, B||C, D
            "end_ns": i64(2000, 4000, 4000, 6000),
            "device_id": i64(0, 0, 0, 0),
            "context_id": i64(1, 1, 1, 1),
            "stream_id": i64(7, 7, 7, 7),
            "correlation_id": i64(corr, corr, corr, corr),
            "graph_id": i64(1, 1, 1, 1),
            "graph_node_id": i64(A, B, C, D),
            "name": np.array(["A", "B", "C", "D"], dtype=object),
            "annotation": np.array([None] * 4, dtype=object),
            "grid_x": i64(1, 1, 1, 1),
            "grid_y": i64(1, 1, 1, 1),
            "grid_z": i64(1, 1, 1, 1),
            "block_x": i64(1, 1, 1, 1),
            "block_y": i64(1, 1, 1, 1),
            "block_z": i64(1, 1, 1, 1),
            "registers_per_thread": i64(0, 0, 0, 0),
            "static_shared_memory": i64(0, 0, 0, 0),
            "dynamic_shared_memory": i64(0, 0, 0, 0),
            "priority": i64(0, 0, 0, 0),
            "queued": i64(0, 0, 0, 0),
            "channel": i64(0, 0, 0, 0),
            "channel_type": i64(0, 0, 0, 0),
        }
        window = {
            "columns": {"kernel": kernel},
            "user_annotations": {},
            "graph_deps": {B: [A], C: [A], D: [B, C]},
        }
        _, events = _trace_window_entries(window, base_ns=0)
        # Launch flow = "f" events keyed by the correlation id; only the root A (ts 1.0us) keeps it.
        launch_ts = [
            e["ts"] for e in events if e.get("ph") == "f" and e.get("id") == corr
        ]
        self.assertEqual(launch_ts, [1.0])
        # The four dep edges are drawn as arrows (1<<40 flow-id base): A->B, A->C, B->D, C->D.
        dep_starts = [
            e for e in events if e.get("ph") == "s" and e.get("id", 0) >= (1 << 40)
        ]
        self.assertEqual(len(dep_starts), 4)
        # Deps OFF: every kernel keeps its launch flow (no suppression).
        window["graph_deps"] = {}
        _, ev_off = _trace_window_entries(window, base_ns=0)
        off = [e for e in ev_off if e.get("ph") == "f" and e.get("id") == corr]
        self.assertEqual(len(off), 4)

    def test_pftrace_render_stages_reassign_lane(self):
        # The pftrace render-stage builder reassigns a graphed kernel onto its logical lane
        # (named by the resolver); a GPU annotation stays on its capture-stream lane (as
        # _window_to_pftrace builds it), not the kernel's reassigned lane. An eager kernel
        # keeps its "stream N" lane. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _assign_render_event_ids,
            _build_render_stages,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        columns = {
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "stream_id": i64(7, 9),  # graphed replayed on stream 7; eager on 9
                "graph_node_id": i64(101, 0),
                "logical_lane": i64(8, 9),  # graphed -> lane 8; eager unchanged
                "lane_name": np.array(["side comms", None], dtype=object),
                "name": np.array(["graphKernel", "eagerKernel"], dtype=object),
            },
            # annotation column as _window_to_pftrace builds it: on its capture stream (7).
            "gpu_annotation": {
                "start_ns": i64(900),
                "end_ns": i64(4100),
                "device_id": i64(0),
                "stream_id": i64(7),
                "name": np.array(["all_reduce"], dtype=object),
            },
        }
        event_id, _corr, wait_off, wait_ids = _assign_render_event_ids(columns, {})
        result = _build_render_stages(
            columns, 1, {}, [], event_id, (wait_off, wait_ids)
        )
        self.assertIsNotNone(result)
        specs, _gfx, stage_cols, *_ = result
        name_by_iid = {iid: name for iid, name, _cat in specs}
        # events follow _RENDER_STAGES order: [graphed kernel, eager kernel, annotation]
        # Lane names carry a zero-padded lane-id prefix (for ordering), then the label.
        lanes = [name_by_iid[int(q)] for q in stage_cols[4].tolist()]
        self.assertTrue(
            lanes[0].endswith("side comms")
        )  # graphed kernel moved to named lane
        self.assertNotEqual(
            lanes[2], lanes[0]
        )  # annotation stays off the reassigned lane
        self.assertIn("stream", lanes[2])  # annotation on its capture-stream lane
        self.assertNotEqual(lanes[1], lanes[0])  # eager kernel is elsewhere
        self.assertIn("stream", lanes[1])  # eager kept its default "stream N" lane

    def test_pftrace_graph_event_and_host_nodes_rendered(self):
        # graph_event_node / graph_host_node are render stages too: an event-record node renders
        # as an "EventRecord" stage and a host-callback node as a "HostNode" stage, both threaded
        # through event-id assignment (so they are dependency-arrow endpoints) and the dependency
        # DAG. One replay (shared correlation): kernel A -> event node -> host node. Only the root
        # (A) links to the launch; the arrows carry the rest. The host callback name/address ride
        # the per-row spread blob (they are string/pointer, not numeric extra_data). No CUDA.
        import json as _json

        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _assign_render_event_ids,
            _build_render_stages,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        gnid_a, gnid_e, gnid_h, corr = 1001, 1002, 1003, 50
        columns = {
            "kernel": {
                "start_ns": i64(1000),
                "end_ns": i64(2000),
                "device_id": i64(0),
                "stream_id": i64(7),
                "correlation_id": i64(corr),
                "graph_node_id": i64(gnid_a),
                "name": np.array(["nodeA"], dtype=object),
            },
            "graph_event_node": {
                "start_ns": i64(2100),
                "end_ns": i64(2100),  # point span
                "device_id": i64(0),
                "stream_id": i64(7),
                "correlation_id": i64(corr),
                "graph_node_id": i64(gnid_e),
            },
            "graph_host_node": {
                "start_ns": i64(2200),
                "end_ns": i64(2300),
                "device_id": i64(0),
                "stream_id": i64(7),
                "correlation_id": i64(corr),
                "graph_node_id": i64(gnid_h),
                "host_fn": np.array(["free"], dtype=object),
                "host_fn_addr": i64(0x1234),
            },
        }
        deps = {
            gnid_e: [gnid_a],
            gnid_h: [gnid_e],
        }  # event waits on A; host waits on event
        event_id, corr_to_eids, wait_off, wait_ids = _assign_render_event_ids(
            columns, deps
        )
        result = _build_render_stages(
            columns, 1, {}, [], event_id, (wait_off, wait_ids)
        )
        self.assertIsNotNone(result)
        specs, _gfx, stage_cols, _extra, _launch, _tables, _panel, meta = result
        name_by_iid = {iid: name for iid, name, _cat in specs}
        self.assertEqual(name_by_iid[5], "EventRecord")
        self.assertEqual(name_by_iid[6], "HostNode")
        # Rows follow _RENDER_STAGES order: [kernel(1), event node(5), host node(6)].
        self.assertEqual(stage_cols[5].tolist(), [1, 5, 6])
        # Only the root A links to the launch; the event/host nodes are reached via arrows.
        self.assertEqual(corr_to_eids[corr].tolist(), [1])
        # Dependency arrows (event_wait_ids CSR): event node waits on A; host on the event node.
        w_off, w_ids = stage_cols[8]

        def waits(i):
            return w_ids[w_off[i] : w_off[i + 1]].tolist()

        self.assertEqual(waits(0), [])  # kernel A is the root
        self.assertEqual(waits(1), [1])  # event node waits on A (event_id 1)
        self.assertEqual(
            waits(2), [2]
        )  # host node waits on the event node (event_id 2)
        # Host callback name/address ride the per-row spread blob (row 2).
        meta_off, meta_bytes = meta
        blob = bytes(meta_bytes[meta_off[2] : meta_off[3]])
        self.assertEqual(_json.loads(blob)["host fn"], "free")
        self.assertEqual(_json.loads(blob)["host fn addr"], hex(0x1234))
        self.assertEqual(meta_off[1] - meta_off[0], 0)  # kernel row: no spread blob

    def test_pftrace_lane_names_order_by_id(self):
        # Perfetto sorts GPU hardware-queue lanes lexicographically by name, so lane names must
        # be prefixed with the zero-padded lane id -> the lane order matches the chrome path's
        # numeric lane-id order. A resolver-named lane must not sort ahead of a lower-id stream
        # lane (regression: "aaa_comm" (lane 50) sorted before "stream 7"). No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _assign_render_event_ids,
            _build_render_stages,
            _HW_QUEUE_IID_BASE,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        columns = {
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "stream_id": i64(7, 40),  # eager on stream 7; graphed replayed on 40
                "graph_node_id": i64(0, 202),
                "logical_lane": i64(7, 50),  # graphed reassigned to lane 50
                # a name that alphabetically sorts before "stream" -- the regression trigger
                "lane_name": np.array([None, "aaa_comm"], dtype=object),
                "name": np.array(["k0", "k1"], dtype=object),
            },
        }
        event_id, _corr, wait_off, wait_ids = _assign_render_event_ids(columns, {})
        specs, *_ = _build_render_stages(
            columns, 1, {}, [], event_id, (wait_off, wait_ids)
        )
        # hw-queue lanes are appended in ascending lane-id order; Perfetto re-sorts by name, so
        # the names sorted lexicographically must equal that ascending-id order.
        lane_names = [name for iid, name, _cat in specs if iid >= _HW_QUEUE_IID_BASE]
        self.assertEqual(len(lane_names), 2)
        self.assertEqual(
            sorted(lane_names), lane_names
        )  # lexicographic == lane-id order
        self.assertIn("stream 7", lane_names[0])  # lane 7 first
        self.assertIn("aaa_comm", lane_names[1])  # lane 50 second, despite the name

    def test_pftrace_annotation_column_from_window(self):
        # pftrace builds its GPU annotation render column from the columnar window (kineto emits
        # no gpu_user_annotation in monitor mode), on the kernels' capture stream not the
        # reassigned lane. Stream 7 keeps eager work, so the span stays there; one whose stream
        # was fully reassigned is dropped (covered by the annotation commit's test). No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import _gpu_annotation_render_column

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        trace_window = {
            "columns": {
                "external_correlation": {
                    "correlation_id": i64(11, 12),
                    "user_external_id": i64(555, 999),
                },
                "kernel": {
                    "correlation_id": i64(11, 12),
                    "device_id": i64(0, 0),
                    "stream_id": i64(7, 7),  # graphed + eager both on capture stream 7
                    "start_ns": i64(1000, 1200),
                    "end_ns": i64(2000, 2200),
                    "graph_node_id": i64(101, 0),  # first graphed, second eager
                    "logical_lane": i64(
                        8, 7
                    ),  # graphed moved to lane 8; eager stays on 7
                },
            },
            "user_annotations": {555: "all_reduce"},
        }
        col = _gpu_annotation_render_column(trace_window, base_ns=0)
        self.assertIsNotNone(col)
        self.assertEqual(col["name"].tolist(), ["all_reduce"])
        self.assertEqual(col["device_id"].tolist(), [0])
        self.assertEqual(
            col["stream_id"].tolist(), [7]
        )  # on the capture stream, not the reassigned lane
        # no annotations in the window -> no column
        self.assertIsNone(
            _gpu_annotation_render_column(
                {"columns": {}, "user_annotations": {}}, base_ns=0
            )
        )

    def test_chrome_counter_events_from_pm(self):
        # PM counters render as chrome "C" (counter) events in a dedicated per-device
        # "GPU N Counters" process row, separate from the GPU kernel work. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _build_chrome_counters,
            _build_pm_counters,
            _gpu_counter_process,
            _merge_counters,
            _pm_label,
        )

        # friendly labels for known metrics; unknown metrics fall back to the raw name
        self.assertEqual(
            _pm_label("nvlrx__bytes.avg.pct_of_peak_sustained_elapsed"), "NVLink RX (%)"
        )
        self.assertEqual(
            _pm_label("some__unknown_metric.sum"), "some__unknown_metric.sum"
        )

        name = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
        n = 4
        pm = {
            "start_ns": np.arange(1000, 1000 + n * 100, 100, dtype=np.int64),
            "device_id": np.zeros(n, dtype=np.int64),
            name: np.linspace(80.0, 95.0, n),
        }
        # empty active_devices -> no device filtering (keep all samples)
        events = _build_chrome_counters(
            _merge_counters(_build_pm_counters(pm, set())), base_ns=0
        )
        counters = [e for e in events if e["ph"] == "C"]
        self.assertEqual(len(counters), n)  # one metric x n samples
        e = counters[0]
        self.assertEqual(e["pid"], _gpu_counter_process(0))  # "GPU 0 Counters"
        self.assertEqual(
            e["name"], "SM Active (%)"
        )  # friendly label for sm__cycles_active
        self.assertEqual(e["ts"], 1.0)  # (1000 - base_ns) / 1000
        self.assertEqual(e["args"], {"": 80.0})
        # counters go on a dedicated string-pid process row so Perfetto renders them as a
        # separate "GPU N Counters" group, not under the device's kernel process
        self.assertTrue(
            any(
                m["ph"] == "M"
                and m["name"] == "process_sort_index"
                and m["pid"] == _gpu_counter_process(0)
                for m in events
            )
        )

    def test_chrome_counter_events(self):
        # GPU counters render as chrome "C" (counter) events in a dedicated per-device
        # "GPU N Counters" process row, separate from the GPU kernel work. No CUDA.
        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _build_chrome_counters,
            _gpu_counter_process,
            _merge_counters,
        )

        name = "Power (W)"
        n = 4
        part = (
            [(1, name)],
            np.zeros(n, dtype=np.int32),  # device (gpu) id
            np.arange(1000, 1000 + n * 100, 100, dtype=np.int64),  # start_ns
            np.ones(n, dtype=np.int32),  # counter_id per sample
            np.linspace(80.0, 95.0, n),  # values
        )
        events = _build_chrome_counters(_merge_counters(part), base_ns=0)
        counters = [e for e in events if e["ph"] == "C"]
        self.assertEqual(len(counters), n)
        e = counters[0]
        self.assertEqual(e["pid"], _gpu_counter_process(0))  # "GPU 0 Counters"
        self.assertEqual(e["name"], name)
        self.assertEqual(e["ts"], 1.0)  # (1000 - base_ns) / 1000
        self.assertEqual(e["args"], {"": 80.0})

    def test_metadata_store_explicit_id(self):
        # put_external(blob, external_id): an explicit non-zero id targets a specific
        # collective rather than the current pushed one -- the seam for a backend to
        # attach metadata outside the push window. external_id 0 (default) still keys
        # by the current id, and the two paths merge. Pure host-side.
        m = torch._C._profiler._cupti_monitor
        m.drain_decoded()  # clear any residue
        try:
            m.metadata_put_external(json.dumps({"backend": "x"}), 77)  # explicit id
        except TypeError:
            self.skipTest("metadata_put_external(blob, external_id) not built yet")
        # Default (0) path keys by the current pushed id and merges with the above.
        m.note_external_push(77)
        m.metadata_put_external(json.dumps({"extra": 1}))
        m.note_external_pop()
        _, ext_meta = m.drain_decoded()
        self.assertEqual(set(ext_meta), {77})
        self.assertEqual(json.loads(ext_meta[77]), {"backend": "x", "extra": 1})

    def test_pftrace_chrome_gpu_slice_parity(self):
        # The .pftrace and chrome-JSON exports are slice-for-slice equivalent for the GPU
        # ops both formats represent (kernels, memcpy, memset, and the GPU-side user
        # annotation). Run the single merge entry point over one synthetic window down both
        # branches and assert the GPU slices agree as a multiset of
        # (name, device, lane, ts_ns, dur_ns): chrome builds event dicts, pftrace builds the
        # native GPU render stages. The pftrace side is compared WITHOUT running (or building)
        # the native encoder: encode_pftrace is stubbed to capture its args, so this stays
        # pure host-side / CI-safe. Legitimate format differences excluded from the compare:
        # chrome also emits CPU ops, ac2g flow (f/s) and metadata (M) events, which the
        # render-stage view represents differently; timestamps are normalized (chrome ts/dur
        # are microsecond floats, render is nanosecond ints -- ns == round(us * 1000)). No CUDA.
        import tempfile

        import numpy as np
        from cupti.cupti import (  # pyrefly: ignore[missing-import]
            Runtime_api_trace_cbid,
        )

        from torch.profiler._cupti.monitor_trace import (
            _runtime_cbid_name,
            merge_trace_window_into_chrome_trace,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        # base_ns 0 so chrome ts/dur (us) map to render ns by *1000 with no base offset.
        base_ns = 0
        # cbid of cudaLaunchKernel (the eager-launch runtime API), for the launch records
        # that produce the CPU-launch -> GPU-op arrows down both export paths.
        launch_cbid = next(
            e.value
            for e in Runtime_api_trace_cbid
            if _runtime_cbid_name(e.value) == "cudaLaunchKernel"
        )
        columns = {
            # eager kernel on stream 7; graphed kernel replayed on stream 7 but reassigned
            # by the resolver to logical lane 8 (named "side comms").
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "context_id": i64(1, 1),
                "stream_id": i64(7, 7),
                "correlation_id": i64(11, 12),
                "graph_id": i64(0, 1),
                "graph_node_id": i64(0, 102),
                "name": np.array(["eagerKernel", "graphKernel"], dtype=object),
                "annotation": np.array([None, None], dtype=object),
                "logical_lane": i64(7, 8),
                "lane_name": np.array([None, "side comms"], dtype=object),
                "grid_x": i64(1, 1),
                "grid_y": i64(1, 1),
                "grid_z": i64(1, 1),
                "block_x": i64(32, 32),
                "block_y": i64(1, 1),
                "block_z": i64(1, 1),
                "registers_per_thread": i64(0, 0),
                "static_shared_memory": i64(0, 0),
                "dynamic_shared_memory": i64(0, 0),
                "priority": i64(0, 0),
                "queued": i64(0, 0),
                "channel": i64(0, 0),
                "channel_type": i64(0, 0),
            },
            # memcpy + memset on stream 9, non-overlapping (so the render-stage per-lane
            # end-clamp is a no-op and durations stay exact for the compare).
            "gpu_memcpy": {
                "start_ns": i64(5000),
                "end_ns": i64(6000),
                "device_id": i64(0),
                "context_id": i64(1),
                "stream_id": i64(9),
                "correlation_id": i64(21),
                "graph_id": i64(0),
                "graph_node_id": i64(0),
                "annotation": np.array([None], dtype=object),
                "bytes": i64(4096),
                "copy_kind": i64(1),
                "src_kind": i64(1),
                "dst_kind": i64(3),
                "flags": i64(0),
                "channel": i64(0),
                "channel_type": i64(0),
            },
            "gpu_memset": {
                "start_ns": i64(7000),
                "end_ns": i64(8000),
                "device_id": i64(0),
                "context_id": i64(1),
                "stream_id": i64(9),
                "correlation_id": i64(22),
                "graph_id": i64(0),
                "graph_node_id": i64(0),
                "annotation": np.array([None], dtype=object),
                "bytes": i64(256),
                "value": i64(0),
                "memory_kind": i64(3),
                "flags": i64(0),
                "channel": i64(0),
                "channel_type": i64(0),
            },
            # eager cudaLaunchKernel records on two distinct CPU threads, correlated to the
            # two kernels (11 -> eager, 12 -> graphed) -> the ac2g / gpu_correlation arrows.
            "cuda_runtime": {
                "start_ns": i64(500, 2500),
                "end_ns": i64(600, 2600),
                "cbid": i64(launch_cbid, launch_cbid),
                "process_id": i64(1000, 1000),
                "thread_id": i64(1, 2),
                "correlation_id": i64(11, 12),
            },
            # only corr 12 (the graphed kernel) maps to a named region -> one GPU annotation.
            "external_correlation": {
                "correlation_id": i64(11, 12),
                "external_id": i64(777, 555),
                "user_external_id": i64(777, 555),
            },
        }
        window = {"columns": columns, "user_annotations": {555: "all_reduce"}}
        gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            cpu = {"baseTimeNanoseconds": base_ns, "traceEvents": []}
            with open(cpu_path, "wb") as f:
                f.write(json.dumps(cpu).encode())

            # chrome side: real merge, collect the GPU X events as (name, dev, lane, ts, dur)ns.
            merge_trace_window_into_chrome_trace(
                cpu_path, os.path.join(d, "out.json"), window
            )
            with open(os.path.join(d, "out.json"), "rb") as f:
                chrome = json.loads(f.read())
            chrome_slices = [
                (
                    e["name"],
                    int(e["pid"]),
                    int(e["tid"]),
                    round(float(e["ts"]) * 1000),
                    round(float(e["dur"]) * 1000),
                )
                for e in chrome["traceEvents"]
                if e.get("ph") == "X" and e.get("cat") in gpu_cats
            ]

            # pftrace side: stub encode_pftrace to capture (name_table, render) and skip the
            # native encode, then reconstruct the same GPU slices from the render stages.
            captured: dict = {}

            def _capture(
                base_ns,
                tracks,
                name_table,
                cat_table,
                group_tuples,
                render,
                counters,
                compression,
            ):
                captured["tracks"] = tracks
                captured["name_table"] = name_table
                captured["cat_table"] = cat_table
                captured["group_tuples"] = group_tuples
                captured["render"] = render
                return b""

            m = torch._C._profiler._cupti_monitor
            with patch.object(m, "encode_pftrace", _capture, create=True):
                merge_trace_window_into_chrome_trace(
                    cpu_path, os.path.join(d, "out.pftrace"), window
                )

            name_table = captured["name_table"]
            specs, _gfx, stage_cols, *_ = captured["render"]
            ts, dur, event_id, gpu, hw_queue_iid, stage, _context, name_iid, _wait = (
                stage_cols
            )
            # specs maps every iid -> name: the stage iids (Kernel/Memcpy/Memset/Annotation)
            # and the hardware-queue lane iids (labeled "[<padded lane id>] <label>").
            name_by_iid = {int(iid): name for iid, name, _cat in specs}
            render_slices = []
            render_lane_names = {}  # (device, lane_id) -> lane label, id prefix stripped
            render_op_lane = {}  # correlation (render event_id) -> (device, lane_id)
            for i in range(len(ts)):
                niid = int(name_iid[i])
                # kernels/annotations carry a name_iid into the global table; memcpy/memset
                # fall back to the stage spec name ("Memcpy"/"Memset").
                name = name_table[niid - 1] if niid else name_by_iid[int(stage[i])]
                label = name_by_iid[int(hw_queue_iid[i])]
                lane_id = int(label[1 : label.index("]")])
                render_lane_names[(int(gpu[i]), lane_id)] = label[
                    label.index("]") + 1 :
                ].strip()
                eid = int(event_id[i])
                if eid:
                    render_op_lane[eid] = (int(gpu[i]), lane_id)
                # render ts is absolute ns; chrome ts was (start - base) / 1000 us.
                render_slices.append(
                    (name, int(gpu[i]), lane_id, int(ts[i]) - base_ns, int(dur[i]))
                )

            # tracks -> (pid, tid) for the CPU launch (thread) tracks: the arrow's source.
            track_pid_tid = {
                t[0]: (int(t[3]), int(t[4])) for t in captured["tracks"] if not t[2]
            }
            # A launch group carries a per-slice gpu_corr CSR (offsets, event_ids) listing the
            # render-stage event_ids it launched (an eager launch -> its one kernel). Build the
            # set of (CPU (pid, tid) source -> (device, lane) target) arrows off those groups,
            # mapping each event_id to its render stage's lane via render_op_lane.
            pftrace_arrows = set()
            for g in captured["group_tuples"]:
                uu, gpu_corr = g[2], g[9]
                if gpu_corr is None:
                    continue
                offsets = np.asarray(gpu_corr[0]).tolist()
                ids = np.asarray(gpu_corr[1]).tolist()
                for slc, u in enumerate(np.asarray(uu).tolist()):
                    for j in range(offsets[slc], offsets[slc + 1]):
                        eid = int(ids[j])
                        pftrace_arrows.add((track_pid_tid[int(u)], render_op_lane[eid]))

        # 2 kernels + memcpy + memset + 1 GPU annotation, identical down both export paths.
        self.assertEqual(len(chrome_slices), 5)
        self.assertEqual(sorted(chrome_slices), sorted(render_slices))

        # Lane-name parity: chrome carries the lane name in thread_name metadata; pftrace in
        # the hw-queue spec label ("[<id>] <name>"). Every lane with GPU slices must agree
        # after stripping pftrace's id prefix and chrome's trailing space -- so the reassigned
        # lane 8 is "side comms" and the default lanes are "stream N" in both.
        gpu_lanes = {(dev, lane) for _n, dev, lane, _t, _d in render_slices}
        chrome_lane_names = {
            (int(e["pid"]), int(e["tid"])): str(e["args"]["name"]).strip()
            for e in chrome["traceEvents"]
            if e.get("ph") == "M" and e.get("name") == "thread_name"
        }
        self.assertEqual(
            {k: chrome_lane_names[k] for k in gpu_lanes},
            {k: render_lane_names[k] for k in gpu_lanes},
        )

        # Arrow (flow) parity: the CPU-launch -> GPU-op link, both endpoints. Chrome emits
        # ac2g flow events -- ph s on the launching CPU thread (source), ph f on the GPU op's
        # lane (target), keyed by id == correlation. pftrace carries the same link as
        # gpu_correlation: the launch track_event's gpu_corr == correlation (source track ->
        # CPU pid/tid) and the GPU render stage's event_id == correlation (target lane).
        # Compare, per correlation, ((cpu_pid, cpu_tid), (gpu_device, gpu_lane)) across formats
        # -- this confirms the arrow connects the same CPU op to the same GPU op in both.
        flow_src = {
            int(e["id"]): (int(e["pid"]), int(e["tid"]))
            for e in chrome["traceEvents"]
            if e.get("cat") == "ac2g" and e.get("ph") == "s"
        }
        flow_tgt = {
            int(e["id"]): (int(e["pid"]), int(e["tid"]))
            for e in chrome["traceEvents"]
            if e.get("cat") == "ac2g" and e.get("ph") == "f"
        }
        chrome_arrows = {
            c: (flow_src[c], flow_tgt[c]) for c in flow_src if c in flow_tgt
        }
        # both launches (corr 11 -> lane 7, 12 -> lane 8) link; not the un-launched memcpy/memset.
        self.assertEqual(set(chrome_arrows), {11, 12})
        self.assertEqual(set(chrome_arrows.values()), pftrace_arrows)

    def test_pftrace_chrome_lane_order_parity(self):
        # The GPU lane DISPLAY ORDER matches across formats -- both order by numeric lane id.
        # Chrome orders lanes by tid (numeric); pftrace names its hardware-queue lanes
        # "[<zero-padded id>] <label>" so that Perfetto's lexicographic sort of those labels
        # reproduces the numeric id order. Assert the id sequences agree and are strictly
        # ascending. The window mixes a low-id "stream" lane (7), a higher-id resolver-named
        # lane (17, "side comms"), and a mid "stream" lane (9): the named high-id lane must NOT
        # sort ahead of the low-id stream lane (the regression the id prefix fixes), and the
        # single- vs double-digit pair (7 vs 17) locks the zero-padding -- without it "[17]"
        # would sort before "[7]". No CUDA.
        import tempfile

        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            _HW_QUEUE_IID_BASE,
            merge_trace_window_into_chrome_trace,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        columns = {
            "kernel": {
                "start_ns": i64(1000, 3000, 5000),
                "end_ns": i64(2000, 4000, 6000),
                "device_id": i64(0, 0, 0),
                "context_id": i64(1, 1, 1),
                "stream_id": i64(7, 7, 9),  # graphed k1 replayed on stream 7
                "correlation_id": i64(11, 12, 13),
                "graph_id": i64(0, 1, 0),
                "graph_node_id": i64(0, 202, 0),
                "name": np.array(["k0", "k1", "k2"], dtype=object),
                "annotation": np.array([None, None, None], dtype=object),
                "logical_lane": i64(7, 17, 9),  # k1 reassigned to lane 17
                "lane_name": np.array([None, "side comms", None], dtype=object),
                "grid_x": i64(1, 1, 1),
                "grid_y": i64(1, 1, 1),
                "grid_z": i64(1, 1, 1),
                "block_x": i64(1, 1, 1),
                "block_y": i64(1, 1, 1),
                "block_z": i64(1, 1, 1),
                "registers_per_thread": i64(0, 0, 0),
                "static_shared_memory": i64(0, 0, 0),
                "dynamic_shared_memory": i64(0, 0, 0),
                "priority": i64(0, 0, 0),
                "queued": i64(0, 0, 0),
                "channel": i64(0, 0, 0),
                "channel_type": i64(0, 0, 0),
            },
        }
        window = {"columns": columns, "user_annotations": {}}
        gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            with open(cpu_path, "wb") as f:
                f.write(
                    json.dumps({"baseTimeNanoseconds": 0, "traceEvents": []}).encode()
                )

            merge_trace_window_into_chrome_trace(
                cpu_path, os.path.join(d, "out.json"), window
            )
            with open(os.path.join(d, "out.json"), "rb") as f:
                chrome = json.loads(f.read())
            # chrome lane order == distinct GPU lanes sorted by tid ascending.
            chrome_order = sorted(
                {
                    int(e["tid"])
                    for e in chrome["traceEvents"]
                    if e.get("ph") == "X" and e.get("cat") in gpu_cats
                }
            )

            captured: dict = {}

            def _capture(
                base_ns,
                tracks,
                name_table,
                cat_table,
                group_tuples,
                render,
                counters,
                compression,
            ):
                captured["render"] = render
                return b""

            m = torch._C._profiler._cupti_monitor
            with patch.object(m, "encode_pftrace", _capture, create=True):
                merge_trace_window_into_chrome_trace(
                    cpu_path, os.path.join(d, "out.pftrace"), window
                )

        specs = captured["render"][0]
        # hw-queue lane specs, sorted lexicographically by label as Perfetto would, then the
        # lane id parsed back out of each "[<padded id>] ..." prefix.
        lane_specs = sorted(
            (name for iid, name, _cat in specs if iid >= _HW_QUEUE_IID_BASE)
        )
        pftrace_order = [int(name[1 : name.index("]")]) for name in lane_specs]

        self.assertEqual(chrome_order, [7, 9, 17])
        self.assertEqual(chrome_order, pftrace_order)
        self.assertEqual(pftrace_order, sorted(pftrace_order))  # strictly ascending
        # zero-padding is load-bearing: labels are "[07]"/"[09]"/"[17]", so lexicographic
        # order == id order (unpadded "[17]" would sort before "[7]").
        self.assertTrue(
            all("]" in name and name.startswith("[0") for name in lane_specs[:2])
        )

    def test_pftrace_categories_and_time_base(self):
        # The pftrace export tags each CPU slice with an interned category (cpu_op /
        # user_annotation / cuda_runtime / ...) so the viewer can total by kind, and
        # threads the trace's base time through to the encoder (-> a ClockSnapshot).
        # encode_pftrace is stubbed to capture its shaped inputs, so this is host-side.
        import tempfile

        import numpy as np
        from cupti.cupti import (  # pyrefly: ignore[missing-import]
            Runtime_api_trace_cbid,
        )

        from torch.profiler._cupti.monitor_trace import (
            _runtime_cbid_name,
            merge_trace_window_into_chrome_trace,
        )

        base_ns = 123_000_000_000
        launch_cbid = next(
            e.value
            for e in Runtime_api_trace_cbid
            if _runtime_cbid_name(e.value) == "cudaLaunchKernel"
        )
        columns = {
            "cuda_runtime": {
                "start_ns": np.array([500], dtype=np.int64),
                "end_ns": np.array([600], dtype=np.int64),
                "cbid": np.array([launch_cbid], dtype=np.int64),
                "process_id": np.array([1000], dtype=np.int64),
                "thread_id": np.array([1], dtype=np.int64),
                "correlation_id": np.array([11], dtype=np.int64),
            },
        }
        window = {"columns": columns, "user_annotations": {}}
        cpu = {
            "baseTimeNanoseconds": base_ns,
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::add",
                    "pid": 1000,
                    "tid": 1,
                    "ts": 1.0,
                    "dur": 5.0,
                },
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "my_region",
                    "pid": 1000,
                    "tid": 1,
                    "ts": 0.0,
                    "dur": 10.0,
                },
            ],
        }

        captured: dict = {}

        def _capture(
            base_ns_arg,
            tracks,
            name_table,
            cat_table,
            group_tuples,
            render,
            counters,
            compression,
        ):
            captured["base_ns"] = base_ns_arg
            captured["name_table"] = name_table
            captured["cat_table"] = cat_table
            captured["group_tuples"] = group_tuples
            return b""

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            with open(cpu_path, "wb") as f:
                f.write(json.dumps(cpu).encode())
            m = torch._C._profiler._cupti_monitor
            with patch.object(m, "encode_pftrace", _capture, create=True):
                merge_trace_window_into_chrome_trace(
                    cpu_path, os.path.join(d, "out.pftrace"), window
                )

        # (2) the trace base time reaches the encoder verbatim (-> ClockSnapshot).
        self.assertEqual(captured["base_ns"], base_ns)

        # (1) every category is interned once and each slice references its own via cat_iid
        # (group tuple index 10), so the (name -> category) mapping is exact.
        name_table = captured["name_table"]
        cat_table = captured["cat_table"]
        self.assertEqual(set(cat_table), {"cpu_op", "user_annotation", "cuda_runtime"})
        pairs = set()
        for gt in captured["group_tuples"]:
            name_iid, cat_iid = gt[3], gt[10]
            if cat_iid is None:
                continue
            for ni, ci in zip(name_iid.tolist(), cat_iid.tolist()):
                self.assertNotEqual(ci, 0)  # every CPU/runtime slice is categorized
                pairs.add((name_table[ni - 1], cat_table[ci - 1]))
        self.assertEqual(
            pairs,
            {
                ("aten::add", "cpu_op"),
                ("my_region", "user_annotation"),
                ("cudaLaunchKernel", "cuda_runtime"),
            },
        )

    def test_pftrace_kernel_carries_collective_metadata(self):
        # The per-kernel collective descriptor (metadata column) is spread onto the GPU
        # render stage as extra_data (name/value string pairs), mirroring the chrome path
        # which spreads the same blob into the GPU op's args -- so consumers need no
        # correlation-id join. Runs the REAL native encoder end-to-end, gunzips, and decodes
        # the protobuf with a hand-rolled scanner (no perfetto python lib). No CUDA. Field
        # numbers (perfetto.h): TracePacket.gpu_render_stage_event=53;
        # GpuRenderStageEvent.extra_data=6; ExtraData.name=1, ExtraData.value=2.
        import tempfile

        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            merge_trace_window_into_chrome_trace,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        descriptor = {"Process Group Name": "0", "dtype": "bf16", "In msg nelems": 1024}
        columns = {
            "kernel": {
                "start_ns": i64(1000),
                "end_ns": i64(2000),
                "device_id": i64(0),
                "context_id": i64(1),
                "stream_id": i64(7),
                "correlation_id": i64(50),
                "graph_id": i64(0),
                "graph_node_id": i64(0),
                "name": np.array(["allreduce_kernel"], dtype=object),
                "annotation": np.array([None], dtype=object),
                "metadata": np.array([json.dumps(descriptor)], dtype=object),
                "grid_x": i64(1),
                "grid_y": i64(1),
                "grid_z": i64(1),
                "block_x": i64(32),
                "block_y": i64(1),
                "block_z": i64(1),
                "registers_per_thread": i64(0),
                "static_shared_memory": i64(0),
                "dynamic_shared_memory": i64(0),
                "priority": i64(0),
                "queued": i64(0),
                "channel": i64(0),
                "channel_type": i64(0),
            },
        }
        window = {"columns": columns, "user_annotations": {}}

        def rv(b, i):  # read a varint at offset i, return (value, next_offset)
            s = r = 0
            while True:
                x = b[i]
                i += 1
                r |= (x & 0x7F) << s
                if not x & 0x80:
                    return r, i
                s += 7

        def flds(b):  # scan a message into [(field_no, wire_type, value)]
            i = 0
            o = []
            while i < len(b):
                k, i = rv(b, i)
                fn, wt = k >> 3, k & 7
                if wt == 0:
                    v, i = rv(b, i)
                    o.append((fn, 0, v))
                elif wt == 2:
                    n, i = rv(b, i)
                    o.append((fn, 2, b[i : i + n]))
                    i += n
                elif wt == 5:
                    i += 4
                elif wt == 1:
                    i += 8
            return o

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            with open(cpu_path, "wb") as f:
                f.write(
                    json.dumps({"baseTimeNanoseconds": 0, "traceEvents": []}).encode()
                )
            out_path = os.path.join(d, "out.pftrace")
            merge_trace_window_into_chrome_trace(cpu_path, out_path, window)
            with gzip.open(out_path, "rb") as f:
                raw = f.read()

        packets = [v for fn, _wt, v in flds(raw) if fn == 1]

        # Collect the extra_data (field 6) name/value pairs across all GPU render stages.
        extra = {}
        for pkt in packets:
            for fn, _wt, v in flds(pkt):
                if fn != 53:
                    continue
                for sfn, _swt, sv in flds(v):
                    if sfn != 6:
                        continue
                    name = value = None
                    for efn, _ewt, ev in flds(sv):
                        if efn == 1:
                            name = bytes(ev).decode()
                        elif efn == 2:
                            value = bytes(ev).decode()
                    if name is not None:
                        extra[name] = value

        # Each descriptor field is spread as its own extra_data entry: JSON strings keep
        # their raw value, numbers are stringified via dump().
        self.assertEqual(extra["Process Group Name"], "0")
        self.assertEqual(extra["dtype"], "bf16")
        self.assertEqual(extra["In msg nelems"], "1024")

    def test_pftrace_render_stage_spreads_graph_annotation(self):
        # A graph-node annotation (mark_kernels region -- e.g. a collective's process-group
        # name recorded at capture via annotate_barrier) is spread onto the GPU render stage as
        # extra_data, the same way the collective metadata column is -- closing the parity gap
        # where chrome spread annotations but the .pftrace render stage dropped them. Also
        # checks annotation + metadata merge. Runs the REAL native encoder, gunzips, decodes
        # the protobuf. No CUDA. Field numbers: gpu_render_stage_event=53, extra_data=6,
        # ExtraData.name=1, ExtraData.value=2.
        import tempfile

        import numpy as np

        from torch.profiler._cupti.monitor_trace import (
            merge_trace_window_into_chrome_trace,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        # annotation column holds the graph annotation resolver's return: a list of dicts per
        # row (as get_kernel_annotations()[node] yields). A second row also carries an eager
        # metadata blob, to check the two channels merge onto one render stage.
        ann = np.empty(2, dtype=object)
        ann[0] = [{"Process Group Name": "5", "In msg nelems": 2048}]
        ann[1] = [{"Process Group Name": "6"}]
        columns = {
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "context_id": i64(1, 1),
                "stream_id": i64(7, 7),
                "correlation_id": i64(50, 51),
                "graph_id": i64(1, 1),
                "graph_node_id": i64((1 << 32) | 10, (1 << 32) | 11),
                "name": np.array(["ar_a", "ar_b"], dtype=object),
                "annotation": ann,
                "metadata": np.array(
                    [None, json.dumps({"dtype": "bf16"})], dtype=object
                ),
                "grid_x": i64(1, 1),
                "grid_y": i64(1, 1),
                "grid_z": i64(1, 1),
                "block_x": i64(32, 32),
                "block_y": i64(1, 1),
                "block_z": i64(1, 1),
                "registers_per_thread": i64(0, 0),
                "static_shared_memory": i64(0, 0),
                "dynamic_shared_memory": i64(0, 0),
                "priority": i64(0, 0),
                "queued": i64(0, 0),
                "channel": i64(0, 0),
                "channel_type": i64(0, 0),
            },
        }
        window = {"columns": columns, "user_annotations": {}}

        def rv(b, i):
            s = r = 0
            while True:
                x = b[i]
                i += 1
                r |= (x & 0x7F) << s
                if not x & 0x80:
                    return r, i
                s += 7

        def flds(b):
            i = 0
            o = []
            while i < len(b):
                k, i = rv(b, i)
                fn, wt = k >> 3, k & 7
                if wt == 0:
                    v, i = rv(b, i)
                    o.append((fn, 0, v))
                elif wt == 2:
                    n, i = rv(b, i)
                    o.append((fn, 2, b[i : i + n]))
                    i += n
                elif wt == 5:
                    i += 4
                elif wt == 1:
                    i += 8
            return o

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            with open(cpu_path, "wb") as f:
                f.write(
                    json.dumps({"baseTimeNanoseconds": 0, "traceEvents": []}).encode()
                )
            out_path = os.path.join(d, "out.pftrace")
            merge_trace_window_into_chrome_trace(cpu_path, out_path, window)
            with gzip.open(out_path, "rb") as f:
                raw = f.read()

        packets = [v for fn, _wt, v in flds(raw) if fn == 1]

        # Per render stage (keyed by event_id), collect its extra_data pairs.
        per_stage: dict[int, dict] = {}
        for pkt in packets:
            for fn, _wt, v in flds(pkt):
                if fn != 53:
                    continue
                event_id = 0
                pairs: dict = {}
                for sfn, _swt, sv in flds(v):
                    if sfn == 1:
                        event_id = sv
                    elif sfn == 6:
                        name = value = None
                        for efn, _ewt, ev in flds(sv):
                            if efn == 1:
                                name = bytes(ev).decode()
                            elif efn == 2:
                                value = bytes(ev).decode()
                        if name is not None:
                            pairs[name] = value
                per_stage[event_id] = pairs

        # Row order (from _render_stage_cols) is stable: row 0 == event_id 1, row 1 == 2.
        # Row 0: annotation-only -> its fields spread (numbers stringified).
        self.assertEqual(per_stage[1]["Process Group Name"], "5")
        self.assertEqual(per_stage[1]["In msg nelems"], "2048")
        # Row 1: annotation + metadata merge onto the one render stage.
        self.assertEqual(per_stage[2]["Process Group Name"], "6")
        self.assertEqual(per_stage[2]["dtype"], "bf16")
        # graphNodeId is emitted with only its lower 32-bit node id; the graph id packed in the
        # upper bits is dropped here (it rides along separately as "graph id").
        self.assertEqual(per_stage[1]["graph node id"], "10")
        self.assertEqual(per_stage[2]["graph node id"], "11")

    def test_pftrace_graph_node_dependency_arrows(self):
        # A CUDA-graph replay's node dependency DAG becomes GpuRenderStageEvent.event_wait_ids
        # (node->node flow arrows) and the cudaGraphLaunch's GpuCorrelation lists only its
        # replay's ROOT node event_ids (the rest are reached through the node->node arrows). Two
        # graphed kernels share one correlation (one replay) with distinct graph_node_ids; B
        # depends on A, so only A (the root) is linked. Runs the REAL native encoder end-to-end,
        # gunzips, and decodes the protobuf with a minimal hand-rolled scanner (the perfetto
        # python lib is not installed). No CUDA. Field numbers verified against perfetto.h:
        # TracePacket.gpu_render_stage_event=53, .track_event=11; GpuRenderStageEvent.event_id=1,
        # .event_wait_ids=18; GpuCorrelation is nested at field 3000 with its
        # render_stage_submission_event_ids at field 1.
        import tempfile

        import numpy as np
        from cupti.cupti import (  # pyrefly: ignore[missing-import]
            Runtime_api_trace_cbid,
        )

        from torch.profiler._cupti.monitor_trace import (
            _runtime_cbid_name,
            merge_trace_window_into_chrome_trace,
        )

        def i64(*vals):
            return np.array(vals, dtype=np.int64)

        base_ns = 0
        graph_launch_cbid = next(
            e.value
            for e in Runtime_api_trace_cbid
            if _runtime_cbid_name(e.value) == "cudaGraphLaunch"
        )
        gnid_a, gnid_b, corr = 1001, 1002, 50
        columns = {
            # Two graphed kernels of ONE replay: same correlation (one launch), distinct
            # graph_node_ids, B depends on A. Same stream (CUPTI piles a replay on one stream).
            "kernel": {
                "start_ns": i64(1000, 3000),
                "end_ns": i64(2000, 4000),
                "device_id": i64(0, 0),
                "context_id": i64(1, 1),
                "stream_id": i64(7, 7),
                "correlation_id": i64(corr, corr),
                "graph_id": i64(1, 1),
                "graph_node_id": i64(gnid_a, gnid_b),
                "name": np.array(["nodeA", "nodeB"], dtype=object),
                "annotation": np.array([None, None], dtype=object),
                "grid_x": i64(1, 1),
                "grid_y": i64(1, 1),
                "grid_z": i64(1, 1),
                "block_x": i64(32, 32),
                "block_y": i64(1, 1),
                "block_z": i64(1, 1),
                "registers_per_thread": i64(0, 0),
                "static_shared_memory": i64(0, 0),
                "dynamic_shared_memory": i64(0, 0),
                "priority": i64(0, 0),
                "queued": i64(0, 0),
                "channel": i64(0, 0),
                "channel_type": i64(0, 0),
            },
            # One cudaGraphLaunch record sharing the replay's correlation -> its GpuCorrelation
            # links only to the root node A (B is reached via the node->node dependency arrow).
            "cuda_runtime": {
                "start_ns": i64(500),
                "end_ns": i64(600),
                "cbid": i64(graph_launch_cbid),
                "process_id": i64(1000),
                "thread_id": i64(1),
                "correlation_id": i64(corr),
            },
        }
        window = {
            "columns": columns,
            "user_annotations": {},
            "graph_deps": {gnid_b: [gnid_a]},  # B depends on A
        }

        def rv(b, i):  # read a varint at offset i, return (value, next_offset)
            s = r = 0
            while True:
                x = b[i]
                i += 1
                r |= (x & 0x7F) << s
                if not x & 0x80:
                    return r, i
                s += 7

        def flds(b):  # scan a message into [(field_no, wire_type, value)]
            i = 0
            o = []
            while i < len(b):
                k, i = rv(b, i)
                fn, wt = k >> 3, k & 7
                if wt == 0:
                    v, i = rv(b, i)
                    o.append((fn, 0, v))
                elif wt == 2:
                    n, i = rv(b, i)
                    o.append((fn, 2, b[i : i + n]))
                    i += n
                elif wt == 5:
                    i += 4
                elif wt == 1:
                    i += 8
            return o

        with tempfile.TemporaryDirectory() as d:
            cpu_path = os.path.join(d, "cpu.json")
            with open(cpu_path, "wb") as f:
                f.write(
                    json.dumps(
                        {"baseTimeNanoseconds": base_ns, "traceEvents": []}
                    ).encode()
                )
            out_path = os.path.join(d, "out.pftrace")
            merge_trace_window_into_chrome_trace(cpu_path, out_path, window)
            with gzip.open(out_path, "rb") as f:
                raw = f.read()

        packets = [v for fn, _wt, v in flds(raw) if fn == 1]

        # Collect the GPU render stages (field 53) as (event_id, [event_wait_ids]).
        render_stages = []
        for pkt in packets:
            for fn, _wt, v in flds(pkt):
                if fn != 53:
                    continue
                event_id = 0
                waits = []
                for sfn, _swt, sv in flds(v):
                    if sfn == 1:
                        event_id = sv
                    elif sfn == 18:
                        waits.append(sv)
                render_stages.append((event_id, waits))

        # Two kernel render stages; exactly one (node B) has event_wait_ids, pointing at the
        # other (node A) -- the node->node dependency arrow.
        self.assertEqual(len(render_stages), 2)
        with_waits = [(e, w) for e, w in render_stages if w]
        self.assertEqual(len(with_waits), 1)
        node_b_event_id, node_b_waits = with_waits[0]
        node_a_event_id = next(e for e, w in render_stages if not w)
        self.assertEqual(node_b_waits, [node_a_event_id])
        self.assertNotEqual(node_a_event_id, node_b_event_id)

        # The cudaGraphLaunch track_event (field 11) carries a GpuCorrelation (nested field
        # 3000) listing only the ROOT node event_id -- node A (render_stage_submission_event_ids
        # at field 1). Node B is not linked from the launch; it is reached via its wait arrow.
        submission_id_sets = []
        for pkt in packets:
            for fn, _wt, v in flds(pkt):
                if fn != 11:
                    continue
                for sfn, _swt, sv in flds(v):
                    if sfn == 3000:
                        ids = [gv for gfn, _gwt, gv in flds(sv) if gfn == 1]
                        submission_id_sets.append(set(ids))
        self.assertEqual(len(submission_id_sets), 1)
        self.assertEqual(submission_id_sets[0], {node_a_event_id})


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiMonitorCUDA(TestCase):
    """Collection through CuptiMonitor directly (not via torch.profiler.profile)."""

    def setUp(self):
        # CuptiMonitor is a process-wide singleton; drop it so each test builds a fresh one.
        from torch.profiler._cupti import monitor as _cupti_monitor

        _cupti_monitor._reset_for_test()
        self.addCleanup(_cupti_monitor._reset_for_test)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_fence_enables_sync_transiently(self):
        # flush(sync=True) fences at a SYNCHRONIZATION sync point, enabled only for
        # the fence (even when no observer requested it) and disabled again after.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        sync = int(ActivityKind.SYNCHRONIZATION)
        monitor = CuptiMonitor()
        obs = monitor.register(
            {ActivityKind.CONCURRENT_KERNEL: {Kernel.END}}, lambda c: None
        )
        self.addCleanup(monitor.unregister, obs)
        self.assertNotIn(sync, monitor._enabled)

        x = torch.randn(64, 64, device="cuda")
        (x @ x).relu().sum().item()
        start = time.time()
        monitor.flush(sync=True)
        self.assertLess(time.time() - start, 2.0)
        self.assertNotIn(sync, monitor._enabled)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH_TOOLS_ID,
        "requires cudaGraphNodeGetToolsId (cuda-bindings >= 13.1 and CUDA driver >= 13.1)",
    )
    def test_event_node_recorder_arm_before_capture(self):
        # End-to-end usage, and the ordering contract it depends on: the recorder learns a
        # graph's ordered event-record nodes from a graph-INSTANTIATE hook, so it has to be
        # armed before the graph is captured. Arming afterwards is too late -- that graph's
        # instantiate already fired, it is never recorded, and every CUDA_EVENT record from
        # its launches then resolves to None (resolve_window refuses to guess).
        def capture_graph_with_event_node():
            x = torch.randn(8, 8, device="cuda")
            # external=True is what makes capture emit an event-record NODE. A default
            # (internal) event is folded into a plain cross-stream dependency edge and
            # produces no node at all, so the bridge would have nothing to resolve. This
            # is the same external form NCCL uses under NCCL_GRAPH_MIXING_SUPPORT=1.
            ev = torch.cuda.Event(external=True)
            # Warm up off the default stream before capture (allocator + kernels).
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    (x @ x).relu()
            torch.cuda.current_stream().wait_stream(s)

            # keep_graph=True keeps the template alive so get_graph_data() can read it
            # after capture; it also defers instantiate(), which is what fires the hook.
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g):
                y = x @ x
                ev.record()
                y.relu()
            g.instantiate()
            return g

        r = _fresh_event_node_recorder(self)

        # Captured while unarmed: the hook never ran, so this graph is absent.
        before = capture_graph_with_event_node()
        before_exec_id = before.get_graph_data()["exec_graph_id"]
        self.assertNotIn(before_exec_id, r.graph_event_nodes)

        r.arm()
        self.assertIsNotNone(r._handle)
        r.arm()  # idempotent: a second arm must not stack a duplicate hook

        after = capture_graph_with_event_node()
        data = after.get_graph_data()
        # Assert rather than skip: if capture stops emitting event-record nodes this test
        # must fail loudly, not quietly pass while covering nothing.
        self.assertTrue(
            any(n["node_type"] == "event_record" for n in data["nodes"]),
            "external event did not produce an event_record node",
        )

        exec_id = data["exec_graph_id"]
        self.assertIn(exec_id, r.graph_event_nodes)
        ordered = r.graph_event_nodes[exec_id]
        # Totally ordered -> a list of tools_ids; each carries the exec graph id up top.
        self.assertIsNotNone(ordered)
        self.assertTrue(ordered)
        for tools_id in ordered:
            self.assertEqual(tools_id >> 32, exec_id)

        # The earlier graph is still absent -- arming does not retroactively record it.
        self.assertNotIn(before_exec_id, r.graph_event_nodes)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH_TOOLS_ID,
        "requires cudaGraphNodeGetToolsId (cuda-bindings >= 13.1 and CUDA driver >= 13.1)",
    )
    def test_event_nodes_on_concurrent_branches_use_node_id_order(self):
        # The serial-chain case above is the one where every candidate ordering agrees. This
        # is the case that separates them: two event nodes on CONCURRENT branches of differing
        # depth. The recorder must report them in node-id (creation) order, which is what
        # cuda_event_sync_id follows -- ordering by dependency depth inverts them here, and
        # does so silently because the two depths are distinct.
        x = torch.randn(8, 8, device="cuda")
        s_deep, s_shallow = torch.cuda.Stream(), torch.cuda.Stream()
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                (x @ x).relu()
        torch.cuda.current_stream().wait_stream(warm)

        ev_deep = torch.cuda.Event(external=True)
        ev_shallow = torch.cuda.Event(external=True)
        r = _fresh_event_node_recorder(self)
        r.arm()

        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            y = x @ x
            s_deep.wait_stream(torch.cuda.current_stream())
            s_shallow.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_deep):  # longer branch, event created FIRST
                a = (y * 2).relu()
                a = (a @ a).relu()
                ev_deep.record()
            with torch.cuda.stream(s_shallow):  # shorter branch, event created SECOND
                b = y + 1
                ev_shallow.record()
            torch.cuda.current_stream().wait_stream(s_deep)
            torch.cuda.current_stream().wait_stream(s_shallow)
            (a + b).sum()
        g.instantiate()

        data = g.get_graph_data()
        nodes = data["nodes"]
        ev_nodes = [n for n in nodes if n["node_type"] == "event_record"]
        self.assertEqual(len(ev_nodes), 2, "expected one event node per branch")

        # get_graph_data() returns nodes in increasing node id -- the invariant the recorder
        # relies on to skip sorting entirely.
        ids = [n["tools_id"] for n in nodes]
        self.assertEqual(ids, sorted(ids))

        recorded = r.graph_event_nodes[data["exec_graph_id"]]
        self.assertEqual(recorded, [n["tools_id"] for n in ev_nodes])
        self.assertEqual(recorded, sorted(recorded))

        # And the branches really are at different dependency depths, so a depth sort would
        # have produced a different (wrong) order here rather than coincidentally agreeing.
        by_index = {n["tools_id"]: i for i, n in enumerate(nodes)}
        depth: dict[int, int] = {}

        def node_depth(i):
            if i in depth:
                return depth[i]
            deps = nodes[i]["dependencies"]
            depth[i] = 0 if not deps else 1 + max(node_depth(j) for j in deps)
            return depth[i]

        d = [node_depth(by_index[t]) for t in recorded]
        self.assertNotEqual(d[0], d[1], "branches should differ in depth")
        if d[0] > d[1]:
            self.assertNotEqual(
                recorded, [t for _, t in sorted(zip(d, recorded))]
            )  # depth order != node-id order

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_v2_columnar_collection(self):
        # End-to-end columnar collection: the monitor turns on a per-activity field
        # selection, decodes each buffer against CUPTI's captured layout, and hands
        # the observer the columns for its selection.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kind = ActivityKind.CONCURRENT_KERNEL
        want = {kind: {Kernel.START, Kernel.END, Kernel.CORRELATION_ID, Kernel.NAME}}

        lock = threading.Lock()
        columns: list = []
        # Fully caller-driven (both periods < 0): no background self-flush and no
        # background drain, so the explicit flush(sync=True) below is the sole,
        # deterministic flush + drain -- nothing races it to consume the decoded columns.
        from torch.profiler._cupti import monitor as cupti_monitor

        cupti_monitor.configure(
            background_flush_period_s=-1, background_drain_period_s=-1
        )
        monitor = CuptiMonitor()

        def on_columns(cols):
            if kind in cols:
                with lock:
                    columns.append(cols[kind])

        try:
            obs = monitor.register(want, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(monitor.unregister, obs)
        self.assertIsNotNone(monitor._subscriber)

        x = torch.randn(256, 256, device="cuda")
        for _ in range(4):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        monitor.flush(sync=True)
        monitor.unregister(obs)

        total = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertGreater(total, 0)
        for c in columns:
            for fld in want[kind]:
                self.assertIn(int(fld), c)
            start = c[int(Kernel.START)]
            end = c[int(Kernel.END)]
            name = c[int(Kernel.NAME)]
            self.assertEqual(len(start), len(end))
            self.assertEqual(len(start), len(name))
            self.assertTrue(all(int(e) - int(s) >= 0 for s, e in zip(start, end)))
        self.assertTrue(any(len(n) > 0 for c in columns for n in c[int(Kernel.NAME)]))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_worker_loop_self_flush_delivers_without_caller_flush(self):
        # worker_loop() with a flush period set (self_flush): the native decode thread
        # drives cuptiActivityFlushAll on its cadence and decodes the completed buffers
        # itself, so records surface with NO caller flush(). Background drain is off, so
        # the only Python work is an explicit drain -- flush() is never called.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kind = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        columns: list = []
        # A real (spread-out) cadence, not a tight spin: the decode thread's first flush is
        # backdated to fire at worker start -- before these kernels exist -- so the kernels'
        # buffers are only delivered by the NEXT cadence flush ~period later. Records showing
        # up therefore proves the periodic self-flush actually fires on schedule.
        flush_period_s = 5.0
        cupti_monitor.configure(
            background_flush_period_s=flush_period_s, background_drain_period_s=-1
        )
        monitor = CuptiMonitor()

        def on_columns(cols):
            if kind in cols:
                with lock:
                    columns.append(cols[kind])

        try:
            obs = monitor.register({kind: {Kernel.START, Kernel.END}}, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(monitor.unregister, obs)

        x = torch.randn(128, 128, device="cuda")
        for _ in range(4):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        # Poll past one cadence: the decode thread self-flushes+decodes on its period
        # (buffers_completed grows with no flush()); drain pulls the decoded columns to the
        # observer. Deadline comfortably exceeds the period so the cadence flush lands.
        deadline = time.time() + flush_period_s * 3 + 5.0
        total = 0
        while time.time() < deadline:
            monitor._drain_and_dispatch()
            with lock:
                total = sum(len(c[int(Kernel.START)]) for c in columns)
            if total > 0:
                break
            time.sleep(0.1)
        self.assertGreater(monitor.stats()["buffers_completed"], 0)
        self.assertGreater(total, 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_worker_loop_no_self_flush_waits_for_caller_flush(self):
        # worker_loop() with no flush period (self_flush off): the decode thread blocks on
        # get_completed() and never self-flushes, so with a normal buffer nothing is
        # delivered until the caller drives flush(). Both cadences off (caller-driven).
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kind = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        columns: list = []
        cupti_monitor.configure(
            background_flush_period_s=-1, background_drain_period_s=-1
        )
        monitor = CuptiMonitor()

        def on_columns(cols):
            if kind in cols:
                with lock:
                    columns.append(cols[kind])

        try:
            obs = monitor.register({kind: {Kernel.START, Kernel.END}}, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(monitor.unregister, obs)

        x = torch.randn(128, 128, device="cuda")
        for _ in range(4):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        # No self-flush and no caller flush: the idle decode thread decodes nothing.
        time.sleep(0.2)
        monitor._drain_and_dispatch()
        with lock:
            before = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertEqual(monitor.stats()["buffers_completed"], 0)
        self.assertEqual(before, 0)

        # The caller-driven flush is what delivers the records.
        monitor.flush(sync=True)
        with lock:
            after = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertGreater(monitor.stats()["buffers_completed"], 0)
        self.assertGreater(after, 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_approx_timestamp_callback_engages_under_udr(self):
        # use_approx hands CUPTI a per-subscriber timestamp callback so records ride the
        # profiler's approx clock. Runs in a child that subscribes before any CUDA context
        # (use_approx is snapshotted at singleton construction). Assert the callback engaged and
        # device records land on the approx clock (< 1e17), not CLOCK_REALTIME (~1e18).
        script = textwrap.dedent(
            """
            import torch
            from cupti.cupti import ActivityKind
            from torch.profiler._cupti import monitor as mon
            from torch.profiler._cupti.records import Kernel

            assert not torch.cuda.is_initialized()
            kind = ActivityKind.CONCURRENT_KERNEL
            seen = []
            mon.configure(use_approx_timestamps=True)
            m = mon.CuptiMonitor()
            obs = m.register({kind: {Kernel.START, Kernel.END}},
                             lambda cols: seen.append(cols[kind]) if kind in cols else None)
            assert m._timestamp_callback_active, "callback did not engage"
            x = torch.randn(256, 256, device="cuda")
            for _ in range(4):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            m.flush(sync=True)
            m.unregister(obs)
            starts = [int(v) for c in seen for v in c[int(Kernel.START)]]
            assert starts, "no kernel records"
            assert max(starts) < 1e17, f"device records off the approx clock: {max(starts)}"
            print("OK", len(starts))
            """
        )
        p = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        self.assertEqual(
            p.returncode, 0, f"child failed:\nstdout={p.stdout}\nstderr={p.stderr}"
        )
        self.assertIn("OK", p.stdout)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_approx_timestamp_callback_engages_with_existing_context(self):
        # With a CUDA context already established, use_approx must still re-time DEVICE records
        # onto the approx clock. Reset the singleton so use_approx (snapshotted at construction)
        # takes effect, establish a context first, then assert the callback engaged and device
        # records land on the approx clock (< 1e17), not CLOCK_REALTIME (~1e18).
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import CuptiError, pylibcupti
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        if pylibcupti().get_version() < 130303:
            self.skipTest(
                "libcupti cannot re-time device records with an existing context"
            )

        cupti_monitor._reset_for_test()
        self.addCleanup(cupti_monitor._reset_for_test)

        # Establish a CUDA context first: this is the pre-existing-context case.
        torch.randn(8, device="cuda").sum().item()
        torch.cuda.synchronize()
        self.assertTrue(torch.cuda.is_initialized())

        kind = ActivityKind.CONCURRENT_KERNEL
        seen: list = []
        cupti_monitor.configure(use_approx_timestamps=True)
        m = CuptiMonitor()

        def on_columns(cols):
            if kind in cols:
                seen.append(cols[kind])

        try:
            obs = m.register({kind: {Kernel.START, Kernel.END}}, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.assertTrue(m._timestamp_callback_active, "callback did not engage")
        try:
            x = torch.randn(256, 256, device="cuda")
            for _ in range(4):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            m.flush(sync=True)
        finally:
            m.unregister(obs)

        starts = [int(v) for c in seen for v in c[int(Kernel.START)]]
        self.assertTrue(starts, "no kernel records")
        self.assertLess(
            max(starts), 1e17, f"device records off the approx clock: {max(starts)}"
        )

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_noisy_apis_suppressed_at_collection(self):
        # disable_noisy_runtime_apis / disable_noisy_driver_apis (run inside activity_enable)
        # should keep the pure-noise cbids out of the records. Assert 0 records for them while
        # kernels + other records still flow (so the check is not vacuous).
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import (
            _noisy_driver_cbids,
            _noisy_runtime_cbids,
            CuptiError,
            pylibcupti,
        )
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Api, Kernel

        if pylibcupti().get_version() < 130303:
            self.skipTest("libcupti predates the UDR per-cbid disable fix")

        noisy_runtime = set(_noisy_runtime_cbids())
        noisy_driver = set(_noisy_driver_cbids())
        self.assertTrue(
            noisy_runtime, "no noisy runtime cbids resolved from the cupti enum"
        )

        runtime_seen: dict[int, int] = {}
        driver_seen: dict[int, int] = {}
        kernels = 0

        def cb(cols):
            nonlocal kernels
            for kind, seen in (
                (ActivityKind.RUNTIME, runtime_seen),
                (ActivityKind.DRIVER, driver_seen),
            ):
                col = cols.get(kind)
                if col and int(Api.CBID.id) in col:
                    for v in col[int(Api.CBID.id)]:
                        seen[int(v)] = seen.get(int(v), 0) + 1
            kk = cols.get(ActivityKind.CONCURRENT_KERNEL)
            if kk:
                kernels += len(next(iter(kk.values())))

        cupti_monitor.configure(buffer_size=1 << 20)
        m = CuptiMonitor()
        want = {
            ActivityKind.CONCURRENT_KERNEL: {Kernel.START, Kernel.END},
            ActivityKind.RUNTIME: {Api.CBID},
            ActivityKind.DRIVER: {Api.CBID},
        }
        try:
            obs = m.register(want, cb)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            x = torch.randn(256, 256, device="cuda")
            for _ in range(50):
                x = torch.relu(
                    x @ x
                )  # each op calls cudaGetDevice/GetLastError internally
            x.sum().item()
            torch.cuda.synchronize()
            m.flush(sync=True)
        finally:
            m.unregister(obs)

        # Records must actually be flowing, else the assertions below are vacuous.
        self.assertGreater(kernels, 0, "no kernel records -- monitor not collecting")
        self.assertGreater(
            sum(runtime_seen.values()), 0, "no RUNTIME records collected"
        )

        leaked_rt = {c: n for c, n in runtime_seen.items() if c in noisy_runtime}
        self.assertEqual(
            leaked_rt, {}, f"noisy runtime cbids not suppressed: {leaked_rt}"
        )
        # DRIVER records only flow on some stacks (e.g. NCCL/driver launches); assert only when
        # driver records are present so the check stays meaningful rather than vacuous.
        if driver_seen:
            leaked_dr = {c: n for c, n in driver_seen.items() if c in noisy_driver}
            self.assertEqual(
                leaked_dr, {}, f"noisy driver cbids not suppressed: {leaked_dr}"
            )

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_singleton_flush_accessible(self):
        # A user can reach the process-wide monitor singleton through CuptiMonitor()
        # and flush it: CuptiMonitor() constructs/returns the one instance, and
        # flush(sync=True) on it delivers everything collected up to the call.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.records import Kernel

        kernel = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        columns: list = []

        def on_columns(cols):
            if kernel in cols:
                with lock:
                    columns.append(cols[kernel])

        mon = cupti_monitor.CuptiMonitor()
        self.assertIs(cupti_monitor.CuptiMonitor(), mon)
        # Drop the singleton after the observer is torn down so the next CuptiMonitor()
        # caller gets a fresh monitor (cleanups run LIFO: unregister first).
        self.addCleanup(setattr, cupti_monitor, "_instance", None)
        try:
            obs = mon.register({kernel: {Kernel.START, Kernel.END}}, on_columns)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        self.addCleanup(mon.unregister, obs)

        x = torch.randn(128, 128, device="cuda")
        for _ in range(3):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()

        # Flush via the singleton fetched from the public accessor, not the local
        # handle, to exercise the user-visible path.
        cupti_monitor.CuptiMonitor().flush(sync=True)

        total = sum(len(c[int(Kernel.START)]) for c in columns)
        self.assertGreater(total, 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_multiple_observers(self):
        # The monitor is the multiplexer: it enables the union of fields on its one
        # subscriber, then hands each observer only the columns it selected. Two
        # observers on the same kind with disjoint selections each see only their own
        # slice (plus KIND id 0) and the same set of records.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kernel = ActivityKind.CONCURRENT_KERNEL
        lock = threading.Lock()
        a_slices: list = []
        b_slices: list = []

        def collect(sink):
            def cb(cols):
                kc = cols.get(kernel)
                if kc:
                    with lock:
                        sink.append({fid: len(col) for fid, col in kc.items()})

            return cb

        monitor = CuptiMonitor()
        try:
            obs_a = monitor.register(
                {kernel: {Kernel.START, Kernel.END}}, collect(a_slices)
            )
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        obs_b = monitor.register(
            {kernel: {Kernel.CORRELATION_ID, Kernel.NAME}}, collect(b_slices)
        )
        self.addCleanup(monitor.unregister, obs_b)
        self.addCleanup(monitor.unregister, obs_a)
        self.assertGreaterEqual(
            set(monitor._enabled.get(int(kernel), frozenset())),
            {0, int(Kernel.START), int(Kernel.CORRELATION_ID)},
        )

        x = torch.randn(128, 128, device="cuda")
        for _ in range(3):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()
        monitor.flush(sync=True)

        self.assertTrue(a_slices)
        self.assertTrue(b_slices)
        a_fields = set().union(*(set(s) for s in a_slices))
        b_fields = set().union(*(set(s) for s in b_slices))
        self.assertLessEqual(a_fields, {0, int(Kernel.START), int(Kernel.END)})
        self.assertLessEqual(
            b_fields, {0, int(Kernel.CORRELATION_ID), int(Kernel.NAME)}
        )
        a_count = sum(s[int(Kernel.START)] for s in a_slices)
        b_count = sum(s[int(Kernel.CORRELATION_ID)] for s in b_slices)
        self.assertGreater(a_count, 0)
        self.assertEqual(a_count, b_count)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_small_buffer_chain_no_deadlock(self):
        # A tiny buffer maximizes buffer-completion churn: the decode worker runs
        # constantly and the foreground + background (20ms) flushes drain, dispatch,
        # and GC the chain back-to-back -- where a flush/decode/lock deadlock would
        # surface. Drive chain push/pop (incl. nested) through it; the test completing
        # is the no-deadlock assertion (a regression would hang here), and we also
        # check the pipeline actually delivered. stop() (final sync flush + decoder
        # teardown) on unregister must not hang either.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Api, ExternalCorrelation, Kernel

        counts: dict[int, int] = {}

        def cb(cols):
            for k, c in cols.items():
                counts[int(k)] = counts.get(int(k), 0) + (
                    len(next(iter(c.values()))) if c else 0
                )

        from torch.profiler._cupti import monitor as cupti_monitor

        cupti_monitor.configure(buffer_size=1024, background_flush_period_s=0.02)
        m = CuptiMonitor()
        want = {
            ActivityKind.CONCURRENT_KERNEL: {
                Kernel.START,
                Kernel.END,
                Kernel.CORRELATION_ID,
                Kernel.GRAPH_NODE_ID,
                Kernel.NAME,
            },
            ActivityKind.EXTERNAL_CORRELATION: {
                ExternalCorrelation.EXTERNAL_ID,
                ExternalCorrelation.CORRELATION_ID,
            },
            ActivityKind.RUNTIME: {Api.CORRELATION_ID},
            ActivityKind.DRIVER: {Api.CORRELATION_ID},
        }
        try:
            obs = m.register(want, cb)
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            x = torch.randn(256, 256, device="cuda")
            for i in range(200):
                m.push_external_correlation_id()
                nested = i % 3 == 0
                if nested:
                    m.push_external_correlation_id()
                x = torch.relu(x @ x)
                if nested:
                    m.pop_external_correlation_id()
                m.pop_external_correlation_id()
                if i % 40 == 0:
                    m.flush()  # foreground drain racing the background flush loop
            x.sum().item()
            torch.cuda.synchronize()
            m.flush(sync=True)
            stats = m.stats()
        finally:
            m.unregister(obs)  # stop(): final sync flush + decoder teardown

        self.assertGreater(counts.get(int(ActivityKind.CONCURRENT_KERNEL), 0), 0)
        self.assertGreater(stats["buffers_completed"], 0)


class TestWindowFinalizer(TestCase):
    """Cover-and-finalize loop of WindowFinalizerMixin -- pure Python, no CUDA/CUPTI.
    A fake user supplies a settable native clock and a synthetic record buffer."""

    class _Fake(WindowFinalizerMixin):
        def __init__(self) -> None:
            self._clock = 0
            self._delivered: list[int] = []  # starts the "monitor" has handed over
            self._buf: list[int] = []  # collected, not-yet-finalized record starts
            # (window_id, boundary, [selected starts]) in finalize order
            self.finalized: list[tuple[int, int, list[int]]] = []
            # Huge interval so the poll thread never fires mid-test; we drive
            # _poll_once() by hand for determinism.
            self._init_observation_window(poll_interval_ms=3_600_000)

        def now_record_ns(self) -> int:
            return self._clock

        def deliver(self, *starts: int) -> None:
            self._delivered.extend(starts)

        def _collect_delivered(self, *, sync: bool) -> None:
            self._buf.extend(self._delivered)
            self._delivered.clear()

        def _window_watermark_ns(self) -> int:
            return max(self._buf) if self._buf else -1

        def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
            selected = [s for s in self._buf if s < boundary_ns]
            self._buf = [s for s in self._buf if s >= boundary_ns]
            self.finalized.append((window_id, boundary_ns, selected))

    def test_boundary_defers_until_covered(self):
        w = self._Fake()
        w._clock = 100
        self.assertEqual(w.mark_boundary(), 0)
        w.deliver(10, 20)  # watermark 20 < 100 -> not covered yet
        w._poll_once()
        self.assertEqual(w.finalized, [])
        w.deliver(150)  # 150 >= 100 -> boundary covered
        w._poll_once()
        self.assertEqual(w.finalized, [(0, 100, [10, 20])])
        self.assertEqual(w._buf, [150])  # consumed trimmed, tail kept
        w._stop_observation_window()

    def test_boundaries_finalize_in_order_as_covered(self):
        w = self._Fake()
        w._clock = 100
        w.mark_boundary()
        w._clock = 200
        w.mark_boundary()
        w.deliver(50, 120)  # watermark 120 covers b0(100), not b1(200)
        w._poll_once()
        self.assertEqual(w.finalized, [(0, 100, [50])])
        self.assertEqual(w._buf, [120])
        w.deliver(250)
        w._poll_once()
        self.assertEqual(w.finalized[-1], (1, 200, [120]))
        w._stop_observation_window()

    def test_drain_all_finalizes_remaining_uncovered(self):
        w = self._Fake()
        w._clock = 100
        w.mark_boundary()
        w._clock = 200
        w.mark_boundary()
        w.deliver(50)  # covers neither by watermark
        w._poll_once()
        self.assertEqual(w.finalized, [])
        w._stop_observation_window()  # drain_all -> finalize both regardless of watermark
        self.assertEqual([f[0] for f in w.finalized], [0, 1])

    def test_poll_thread_starts_once_and_stops(self):
        w = self._Fake()
        w._clock = 10
        w.mark_boundary()
        t = w._poll_thread
        self.assertIsNotNone(t)
        w._clock = 20
        w.mark_boundary()
        self.assertIs(w._poll_thread, t)  # not restarted
        w._stop_observation_window()
        self.assertIsNone(w._poll_thread)


@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not TEST_CUDA, "CUDA is required")
class TestCuptiMonitorProfiler(TestCase):
    """The monitor driven through ``torch.profiler.profile`` (trace shape, op/kernel
    parity, record_shapes, sync/async export, multithread thread-assignment, ...)."""

    def setUp(self):
        # CuptiMonitor is a process-wide singleton; drop it so each test builds a fresh one.
        # The class also hosts CUPTI-free config-validation tests, so the monitor module
        # (which needs cupti-python) may be unimportable; there is nothing to reset then.
        if not TEST_CUPTI_PYTHON:
            return
        from torch.profiler._cupti import monitor as _cupti_monitor

        _cupti_monitor._reset_for_test()
        self.addCleanup(_cupti_monitor._reset_for_test)

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    @unittest.skipUnless(
        SM100OrLater, "hardware event sampling requires GB200+ (sm_100)"
    )
    def test_cupti_monitor_enable_hes_early_guard(self):
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
import torch
from torch.profiler._cupti import monitor as _cupti_monitor

_cupti_monitor.enable_hes_early()
assert _cupti_monitor.is_hes_enabled()
""",
            ],
            text=True,
            timeout=60,
        )

        p = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import torch
from torch.profiler._cupti import monitor as _cupti_monitor

torch.randn(1, device="cuda")
_cupti_monitor.enable_hes_early()
""",
            ],
            text=True,
            timeout=60,
            capture_output=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn(
            "enable_hes_early() must be called before CUDA context creation",
            p.stderr,
        )

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_collection_smoke(self):
        from torch.profiler._cupti import monitor as _cupti_monitor
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        obs = NodeTimerObserver()
        self.assertTrue(obs.available)

        x = torch.randn(64, 64, device="cuda")
        y = torch.relu(x + 1)
        y.sum().item()
        torch.cuda.synchronize()

        monitor = _cupti_monitor.CuptiMonitor()
        monitor.flush(sync=True)
        stats = monitor.stats()
        _gnode, start, _end, _stream = obs.drain()
        obs.close()

        # The native C++ pool must actually have been exercised: catches a silent
        # regression to a no-op (e.g. broken callback registration or symbol
        # export) that would still pass if the worker never saw a buffer. The
        # monitor demuxes to columns and the observer drains spans, so real kernel
        # spans (NodeTimerObserver collects CONCURRENT_KERNEL) must come out.
        self.assertGreater(stats["buffers_allocated"], 0)
        self.assertGreater(stats["buffers_completed"], 0)
        self.assertEqual(stats["buffers_pending"], 0)
        self.assertGreater(len(start), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_graph_host_node_collected_on_device(self):
        # End-to-end: a CUDA-graph host node (a CPU callback run as a graph node) is collected
        # as a GRAPH_HOST_NODE record and rendered as a "graph_host_node" span in the exported
        # trace. The graph is captured with torch.cuda.graph (keep_graph so the template stays
        # live), then a host node is grafted onto it via cudaGraphAddHostNode -- stream capture
        # never enqueues a host node, so the one node under test needs the runtime primitive --
        # and the graph is re-instantiated before replay. The node is a libc free(NULL): a safe
        # no-op on every replay. With enable_graph_dependencies (armed before instantiate, as a
        # real early-arm would be), the callback's symbol name is recovered from the graph and
        # the span is named "HostNode: free". Needs a driver new enough to emit the records
        # (CUDA >= 13.2 / cuda-compat); the CUPTI enable succeeds on older drivers but yields
        # nothing, so skip rather than falsely fail.
        import ctypes

        from torch.profiler._cupti._graph_deps import _GraphDependencyRecorder

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            self.skipTest("cuda.bindings required")
        drv = cudart.cudaDriverGetVersion()[1]
        if drv < 13020:
            self.skipTest(f"driver {drv} does not emit GRAPH_HOST_NODE (needs >= 13.2)")

        def ck(ret):
            if int(ret[0]) != int(cudart.cudaError_t.cudaSuccess):
                raise RuntimeError(f"cuda err {ret[0]}")
            return ret[1] if len(ret) > 1 else None

        free_addr = ctypes.cast(ctypes.CDLL(None).free, ctypes.c_void_p).value
        params = cudart.cudaHostNodeParams()
        try:
            params.fn = cudart.cudaHostFn_t(init_value=free_addr)
        except TypeError:
            params.fn = free_addr
        params.userData = 0  # free(NULL): safe no-op on every replay

        # Arm before instantiate so the recorder captures the host node's topology + fn name.
        _GraphDependencyRecorder().arm()
        x = torch.zeros(8, device="cuda")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            x.add_(1.0)
        # raw_cuda_graph() is an int handle; pass it straight to the binding (no typed wrapper).
        ck(cudart.cudaGraphAddHostNode(graph.raw_cuda_graph(), [], 0, params))
        graph.instantiate()

        cfg = _ExperimentalConfig(
            custom_profiler_config='{"backend":"cupti_monitor","enable_graph_dependencies":true}'
        )
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            gz = trace_path + ".gz"
            if os.path.exists(gz):
                with gzip.open(gz, "rt") as f:
                    data = json.load(f)
            else:
                with open(trace_path) as f:
                    data = json.load(f)

        host = [e for e in data["traceEvents"] if e.get("cat") == "graph_host_node"]
        self.assertGreater(len(host), 0)
        self.assertIn("host thread", host[0]["args"])
        # The grafted node is libc free; its symbol is recovered and names the span.
        self.assertEqual(host[0]["name"], "HostNode: free")
        self.assertEqual(host[0]["args"]["host fn"], "free")

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_collection_repeated_lifecycle(self):
        from torch.profiler._cupti import monitor as _cupti_monitor
        from torch.profiler._cupti.observers.node_timer import NodeTimerObserver

        # Register/collect/unregister twice: the last observer leaving stops the
        # monitor, so the second pass exercises the start-after-stop restart path.
        for _ in range(2):
            obs = NodeTimerObserver()
            self.assertTrue(obs.available)

            x = torch.randn(32, 32, device="cuda")
            y = torch.sigmoid(x)
            y.sum().item()
            torch.cuda.synchronize()

            monitor = _cupti_monitor.CuptiMonitor()
            monitor.flush(sync=True)
            _gnode, start, _end, _stream = obs.drain()
            obs.close()

            self.assertGreater(len(start), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_multithread_runtime_thread_assignment(self):
        x1 = torch.randn(256, 256, device="cuda")
        x2 = torch.randn(256, 256, device="cuda")
        y1 = torch.randn(256, 256, device="cuda")
        y2 = torch.randn(256, 256, device="cuda")

        # Warm up kernel/runtime state so the profiled region is dominated by the
        # launches from the two worker threads.
        _ = torch.relu(x1 + y1)
        _ = torch.relu(x2 + y2)
        torch.cuda.synchronize()

        start_evt = threading.Event()

        def worker(name, x, y):
            start_evt.wait()
            with record_function(name):
                z = torch.relu(x + y)
                z.sum().item()
                torch.cuda.synchronize()

        cfg = _ExperimentalConfig(
            profile_all_threads=True,
            custom_profiler_config='{"backend":"cupti_monitor"}',
        )

        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                threads = [
                    threading.Thread(target=worker, args=("worker_a", x1, y1)),
                    threading.Thread(target=worker, args=("worker_b", x2, y2)),
                ]
                for thread in threads:
                    thread.start()
                start_evt.set()
                for thread in threads:
                    thread.join()

            prof.export_chrome_trace(trace_path)
            # cupti_monitor writes the trace gzipped at <path>.gz (synchronously by default).
            gz = trace_path + ".gz"
            if os.path.exists(gz):
                with gzip.open(gz, "rt") as f:
                    data = json.load(f)
            else:
                with open(trace_path) as f:
                    data = json.load(f)

        events = data["traceEvents"]
        worker_tids = sorted(
            {
                e["tid"]
                for e in events
                if e.get("ph") == "X"
                and e.get("cat") == "user_annotation"
                and e.get("name") in {"worker_a", "worker_b"}
                and isinstance(e.get("tid"), int)
            }
        )
        launch_tids = sorted(
            {
                e["tid"]
                for e in events
                if e.get("ph") == "X"
                and e.get("cat") == "cuda_runtime"
                and e.get("name") == "cudaLaunchKernel"
                and isinstance(e.get("tid"), int)
            }
        )

        self.assertEqual(len(worker_tids), 2)
        self.assertGreater(len(launch_tids), 0)
        self.assertTrue(set(launch_tids).issubset(set(worker_tids)))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_runtime_thread_without_record_function(self):
        # Regression: CPU-side CUPTI records (cuda_runtime/cuda_driver) must land on the
        # issuing OS thread -- the same lane as their cpu_ops, matching kineto -- even
        # with NO record_function region. The thread map is otherwise only populated by
        # push_annotation (record_function); open_window captures the starting thread so
        # the bare path works too. A cuBLAS GEMM surfaces CUPTI's raw pthread-style
        # threadId, which without the capture lands on a phantom lane.
        import torch.nn as nn

        mlp = (
            nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))
            .cuda()
            .eval()
        )
        x = torch.randn(32, 512, device="cuda")
        cfg = _ExperimentalConfig(
            custom_profiler_config='{"backend":"cupti_monitor"}',
        )
        with TemporaryFileName(mode="w+") as trace_path:
            with (
                torch.no_grad(),
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    experimental_config=cfg,
                ) as prof,
            ):
                for _ in range(3):
                    mlp(x)
                torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            gz = trace_path + ".gz"
            if os.path.exists(gz):
                with gzip.open(gz, "rt") as f:
                    events = json.load(f)["traceEvents"]
            else:
                with open(trace_path) as f:
                    events = json.load(f)["traceEvents"]

        def tids(cat):
            return {
                e["tid"]
                for e in events
                if e.get("ph") == "X"
                and e.get("cat") == cat
                and isinstance(e.get("tid"), int)
            }

        cpu_tids = tids("cpu_op")
        runtime_tids = tids("cuda_runtime")
        self.assertGreater(len(cpu_tids), 0)
        self.assertGreater(len(runtime_tids), 0)
        # The CPU-side CUPTI records share the issuing thread with the cpu_ops (no
        # phantom CUPTI-threadId lane).
        self.assertTrue(runtime_tids.issubset(cpu_tids))

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_trace_has_expected_events(self):
        cfg = _ExperimentalConfig(
            custom_profiler_config='{"backend":"cupti_monitor"}',
        )
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                with record_function("monitor_region"):
                    a = torch.randn(128, 128, device="cuda")
                    b = torch.randn(128, 128, device="cuda")
                    c = (a @ b).relu()
                    _ = c.cpu()
                    torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            # cupti_monitor writes the trace gzipped at <path>.gz (synchronously by default).
            gz = trace_path + ".gz"
            if os.path.exists(gz):
                with gzip.open(gz, "rt") as f:
                    events = json.load(f)["traceEvents"]
            else:
                with open(trace_path) as f:
                    events = json.load(f)["traceEvents"]

        cats = {e.get("cat") for e in events if e.get("ph") == "X"}
        for expected in (
            "kernel",
            "cuda_runtime",
            "gpu_memcpy",
            "cpu_op",
            "user_annotation",
        ):
            self.assertIn(
                expected,
                cats,
                f"missing {expected}; got {sorted(c for c in cats if c)}",
            )

        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        self.assertGreater(len(kernels), 0)
        self.assertTrue(all(e["dur"] > 0 for e in kernels))

        runtime_names = {
            e.get("name") for e in events if e.get("cat") == "cuda_runtime"
        }
        self.assertIn("cudaLaunchKernel", runtime_names)

        user_names = {
            e["name"]
            for e in events
            if e.get("cat") == "user_annotation" and e.get("ph") == "X"
        }
        self.assertIn("monitor_region", user_names)

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    def test_cupti_monitor_observer_registration_failure_is_graceful(self):
        # If the per-cycle ProfilerObserver fails to register with the CUPTI monitor (an
        # intermittent CUPTI condition), the profiler must degrade gracefully: with no
        # observer / trace window, stop_trace and export_chrome_trace skip the trace instead
        # of asserting and taking down the run.
        from torch.profiler._cupti import monitor as _cupti_monitor

        cfg = _ExperimentalConfig(custom_profiler_config='{"backend":"cupti_monitor"}')
        with patch.object(
            _cupti_monitor.CuptiMonitor,
            "register",
            side_effect=RuntimeError("simulated observer registration failure"),
        ):
            with TemporaryFileName(mode="w+") as trace_path:
                # Exiting the profiler runs stop_trace -- it must not raise even though the
                # observer never registered.
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    experimental_config=cfg,
                ) as prof:
                    a = torch.randn(64, 64, device="cuda")
                    _ = (a @ a).cpu()
                    torch.cuda.synchronize()
                # Registration failed -> observer unavailable, no trace window.
                obs = prof._cupti_profiler_observer
                self.assertTrue(obs is None or not obs.available)
                # Must skip the export rather than assert/crash.
                prof.export_chrome_trace(trace_path)
                prof.wait_for_exports()

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_sync_export_default(self):
        # Default: the cupti_monitor backend exports synchronously -- the merged file is
        # on disk when export_chrome_trace returns, wait_for_exports is a no-op, and no
        # background poll thread is spawned (the finalize runs on the calling thread).
        cfg = _ExperimentalConfig(custom_profiler_config='{"backend":"cupti_monitor"}')
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                a = torch.randn(128, 128, device="cuda")
                (a @ a).relu().sum().item()
                torch.cuda.synchronize()
            obs = prof._cupti_profiler_observer
            self.assertIsNotNone(obs)
            self.assertIsNone(obs._poll_thread)  # sync -> no background poller spawned
            prof.export_chrome_trace(trace_path)
            # Written before return (cupti_monitor writes gzipped at <path>.gz).
            self.assertTrue(
                os.path.exists(trace_path + ".gz") or os.path.exists(trace_path)
            )
            prof.wait_for_exports()  # no-op in sync mode

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_async_export_defers(self):
        # cupti_monitor_async_export=true hands the export off: a background poll thread
        # is spawned and the file is written off-thread, joined by wait_for_exports.
        cfg = _ExperimentalConfig(
            custom_profiler_config=(
                '{"backend":"cupti_monitor","cupti_monitor_async_export":true}'
            ),
        )
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                a = torch.randn(128, 128, device="cuda")
                (a @ a).relu().sum().item()
                torch.cuda.synchronize()
            obs = prof._cupti_profiler_observer
            self.assertIsNotNone(obs._poll_thread)  # async -> background poller running
            prof.export_chrome_trace(trace_path)
            prof.wait_for_exports()
            self.assertTrue(
                os.path.exists(trace_path + ".gz") or os.path.exists(trace_path)
            )

    def test_cupti_monitor_async_export_requires_backend(self):
        # cupti_monitor_async_export is a cupti_monitor-only option: setting it without
        # the backend is a misconfiguration, not a silent no-op. (Pure config
        # validation -- raises at construction, no CUDA/cupti needed.)
        with self.assertRaisesRegex(ValueError, "cupti_monitor_async_export"):
            profile(
                activities=[ProfilerActivity.CPU],
                experimental_config=_ExperimentalConfig(
                    custom_profiler_config='{"cupti_monitor_async_export":false}'
                ),
            )

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @_isolated
    def test_cupti_monitor_record_shapes(self):
        cfg = _ExperimentalConfig(
            custom_profiler_config='{"backend":"cupti_monitor"}',
        )

        def shaped_cpu_ops(record_shapes):
            with TemporaryFileName(mode="w+") as trace_path:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=record_shapes,
                    experimental_config=cfg,
                ) as prof:
                    a = torch.randn(64, 64, device="cuda")
                    (a @ a).relu()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(trace_path)
                # cupti_monitor writes the trace gzipped at <path>.gz (synchronously by default).
                gz = trace_path + ".gz"
                if os.path.exists(gz):
                    with gzip.open(gz, "rt") as f:
                        events = json.load(f)["traceEvents"]
                else:
                    with open(trace_path) as f:
                        events = json.load(f)["traceEvents"]
            return [
                e
                for e in events
                if e.get("cat") == "cpu_op" and "Input Dims" in e.get("args", {})
            ]

        # record_shapes is a CPU-side setting, so it must flow through the monitor
        # backend just like the stock profiler.
        self.assertEqual(shaped_cpu_ops(record_shapes=False), [])
        self.assertGreater(len(shaped_cpu_ops(record_shapes=True)), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_cupti_monitor_matches_stock_op_and_kernel_names(self):
        # Run in a FRESH process. This test needs a stock (Kineto) CUDA baseline and
        # then a cupti_monitor session, so it must start from a process that hasn't
        # touched CUPTI -- immune to whatever earlier tests did to this process's
        # CUPTI. Inside it, stock (Kineto) runs first, then cuptiFinalize() releases
        # CUPTI synchronously (rather than Kineto's async TEARDOWN_CUPTI, whose
        # deferred global finalize races and can deadlock the monitor's teardown) so
        # the following monitor session can subscribe.
        import subprocess

        script = textwrap.dedent(
            """
            import gzip, json, os, tempfile
            import torch
            from torch.profiler import profile, ProfilerActivity
            from torch._C._profiler import _ExperimentalConfig

            def trace_summary(use_monitor):
                cfg = _ExperimentalConfig(
                    custom_profiler_config='{"backend":"cupti_monitor"}'
                    if use_monitor else "")
                with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        experimental_config=cfg,
                    ) as prof:
                        a = torch.randn(128, 128, device="cuda")
                        b = torch.randn(128, 128, device="cuda")
                        (a @ b).relu().sum()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(f.name)
                    # cupti_monitor writes the trace gzipped at <name>.gz (sync by default);
                    # stock writes <name> directly.
                    gz = f.name + ".gz"
                    if os.path.exists(gz):
                        events = json.load(gzip.open(gz))["traceEvents"]
                    else:
                        events = json.load(open(f.name))["traceEvents"]
                aten = {e["name"] for e in events
                        if e.get("cat") == "cpu_op"
                        and e.get("name", "").startswith("aten::")}
                nker = sum(1 for e in events
                           if e.get("cat") == "kernel" and e.get("ph") == "X")
                return aten, nker

            stock_ops, stock_kernels = trace_summary(False)
            # Synchronously release CUPTI from the stock (Kineto) session so the
            # monitor can subscribe -- cuptiFinalize() now, with nothing else using
            # CUPTI, instead of Kineto's async TEARDOWN_CUPTI finalize (which races
            # and can deadlock the monitor's teardown).
            from torch.profiler._cupti.cupti_python import pylibcupti
            pylibcupti().finalize()
            monitor_ops, monitor_kernels = trace_summary(True)
            assert stock_kernels > 0, f"stock kernels={stock_kernels}"
            assert monitor_kernels > 0, f"monitor kernels={monitor_kernels}"
            assert monitor_ops == stock_ops, (
                f"ops differ: only_stock={sorted(stock_ops - monitor_ops)} "
                f"only_monitor={sorted(monitor_ops - stock_ops)}")
            print("OK", stock_kernels, monitor_kernels)
            """
        )
        # The child inherits this process's libcupti (LD_PRELOAD/LD_LIBRARY_PATH) via
        # the environment.
        p = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(
            p.returncode,
            0,
            f"subprocess failed:\nstdout={p.stdout}\nstderr={p.stderr}",
        )
        self.assertIn("OK", p.stdout)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_cupti_monitor_kineto_parity(self):
        # In a FRESH process (clean CUPTI), profile a couple of representative models
        # under stock (Kineto) and then the cupti_monitor backend, eager AND graphed,
        # and assert the aten-op set + kernel-name multiset are identical between the
        # two backends -- the monitor must observe the same ops/kernels Kineto does.
        # Stock runs first; cuptiFinalize() then releases CUPTI so the monitor can
        # subscribe. The child inherits this process's libcupti via the environment.
        import subprocess

        script = textwrap.dedent(
            """
            import gzip, json, os, tempfile
            import torch, torch.nn as nn
            from torch.profiler import profile, ProfilerActivity
            from torch._C._profiler import _ExperimentalConfig

            # HES (hardware events) must be armed before any CUDA context exists; the
            # monitor must still produce kernel metadata identical to Kineto with it on.
            # Arm it first thing and assert no context exists yet -- a failure here means
            # the harness created a context too early (a test bug), distinct from HES
            # being unsupported on the platform (is_hes_enabled stays False -> SKIP_HES).
            HES = os.environ.get("PARITY_HES") == "1"
            if HES:
                assert not torch.cuda.is_initialized(), "CUDA context exists before HES arm"
                from torch.profiler._cupti import monitor as _cupti_monitor
                from torch.profiler._cupti.cupti_python import CuptiError
                try:
                    _cupti_monitor.enable_hes_early()
                except CuptiError as e:
                    # HES is unsupported on this platform (cuptiActivityEnableHWTrace
                    # -> CUPTI_ERROR_NOT_SUPPORTED): skip the HES leg of the parity check.
                    if "CUPTI_ERROR_NOT_SUPPORTED" not in str(e):
                        raise
                    print("SKIP_HES"); raise SystemExit(0)
                if not _cupti_monitor.is_hes_enabled():
                    print("SKIP_HES"); raise SystemExit(0)

            def make_models():
                torch.manual_seed(0)
                mlp = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(),
                                    nn.Linear(4096, 1024), nn.GELU(), nn.Linear(1024, 1024))
                enc = nn.TransformerEncoderLayer(d_model=512, nhead=8,
                        dim_feedforward=2048, batch_first=True, dropout=0.0)
                return {"mlp": (mlp.cuda().eval(), torch.randn(64, 1024, device="cuda")),
                        "transformer": (enc.cuda().eval(),
                                        torch.randn(8, 128, 512, device="cuda"))}

            def capture(model, inp):
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3): model(inp)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): model(inp)
                return g

            def summary(model, inp, mode, use_monitor):
                # The cupti_monitor backend writes the trace gzipped at <path>.gz
                # (synchronously by default), so export then read the .gz.
                g = capture(model, inp) if mode == "graphed" else None
                cfg = _ExperimentalConfig(
                    custom_profiler_config='{"backend":"cupti_monitor"}' if use_monitor else "")
                with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
                    with torch.no_grad(), profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        experimental_config=cfg) as prof:
                        for _ in range(5):
                            g.replay() if mode == "graphed" else model(inp)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(f.name)
                    gz = f.name + ".gz"
                    if os.path.exists(gz):
                        ev = json.load(gzip.open(gz))["traceEvents"]; os.remove(gz)
                    else:
                        ev = json.load(open(f.name))["traceEvents"]
                aten = frozenset(e["name"] for e in ev
                    if e.get("cat") == "cpu_op" and e.get("name", "").startswith("aten::"))
                # name -> multiset of per-kernel launch configs. Excludes timing
                # (start/end/queued), backend-specific ids (correlation/graph), and
                # the occupancy fields kineto derives; the rest must match exactly.
                META = ("grid", "block", "registers per thread", "shared memory",
                        "device", "context", "channel", "channel_type")
                meta = {}
                for e in ev:
                    if e.get("cat") == "kernel" and e.get("ph") == "X":
                        a = e.get("args") or {}
                        key = tuple(tuple(a[k]) if isinstance(a.get(k), list)
                                    else a.get(k) for k in META)
                        d = meta.setdefault(e["name"], {})
                        d[key] = d.get(key, 0) + 1
                # Launch parity (eager): cudaLaunchKernel calls, and the ac2g flow arrows
                # that terminate at a GPU op (kernel/gpu_memset) -- one inbound launch
                # arrow per GPU op in both backends. (Kineto also draws arrows ending at
                # cuda_runtime correlation hops, which we deliberately ignore.)
                gpu_ops = [e for e in ev if e.get("ph") == "X"
                           and e.get("cat") in ("kernel", "gpu_memset")]
                launches = sum(1 for e in ev if e.get("ph") == "X"
                               and e.get("cat") == "cuda_runtime"
                               and e.get("name") == "cudaLaunchKernel")
                def _on_gpu(fe):
                    p, t, ts = fe.get("pid"), fe.get("tid"), fe.get("ts", 0)
                    return any(x.get("pid") == p and x.get("tid") == t
                               and x.get("ts", 0) <= ts <= x.get("ts", 0) + x.get("dur", 0)
                               for x in gpu_ops)
                arrows = sum(1 for e in ev if e.get("ph") in ("f", "t")
                             and e.get("cat") == "ac2g" and _on_gpu(e))
                launch = {"launches": launches, "arrows": arrows, "gpu_ops": len(gpu_ops)}
                return aten, meta, launch

            # Graphed mode profiles CUDA-graph replay, which needs
            # cudaGraphNodeGetToolsId (cuda-compat / new driver): without it node
            # correlation no-ops and, with HES on, cudaDeviceSynchronize wedges.
            # Run eager-only when the tools-id API is unusable.
            from torch.cuda._graph_annotations import _is_tools_id_unavailable
            if _is_tools_id_unavailable():
                print("SKIP_GRAPHED")
                MODES = ("eager",)
            else:
                MODES = ("eager", "graphed")
            models = make_models()

            def nkernels(meta):
                return sum(sum(c.values()) for c in meta.values())

            # HES is process-global and armed before CUDA init, which perturbs a stock
            # Kineto session (it drops activity records). So compare the monitor against
            # stock only with HES off; the HES run instead emits its monitor metadata for
            # the parent to compare against the HES-off monitor (transitively == stock).
            if not HES:
                stock = {(n, mode): summary(m, i, mode, False)
                         for n, (m, i) in models.items() for mode in MODES}
                from torch.profiler._cupti.cupti_python import pylibcupti
                pylibcupti().finalize()
            mon = {(n, mode): summary(m, i, mode, True)
                   for n, (m, i) in models.items() for mode in MODES}

            if not HES:
                for n in models:
                    for mode in MODES:
                        sa, sm, sx = stock[(n, mode)]; ma, mm, mx = mon[(n, mode)]
                        assert nkernels(sm) > 0, f"{n}/{mode}: no stock kernels"
                        assert nkernels(mm) > 0, f"{n}/{mode}: no monitor kernels"
                        assert sa == ma, (f"{n}/{mode} aten differ: "
                            f"only_stock={sorted(sa - ma)} only_mon={sorted(ma - sa)}")
                        # Parity on kernel names, per-kernel launch counts, AND launch
                        # config (grid/block/regs/shared/...) -- everything but timing.
                        assert sm == mm, (f"{n}/{mode} kernel metadata differ: "
                            f"name_only_stock={sorted(set(sm) - set(mm))} "
                            f"name_only_mon={sorted(set(mm) - set(sm))} "
                            f"config_diffs={ {k: (sm.get(k), mm.get(k)) for k in set(sm) & set(mm) if sm[k] != mm[k]} }")
                        if mode == "eager":
                            # One inbound launch arrow per GPU op in both backends, and
                            # cudaLaunchKernel counts match (graphed replays launch via a
                            # single cudaGraphLaunch, so this only holds eager).
                            assert sx["arrows"] == sx["gpu_ops"] == mx["arrows"] == mx["gpu_ops"], (
                                f"{n} launch-arrow parity: stock={sx} monitor={mx}")
                            assert sx["launches"] == mx["launches"], (
                                f"{n} cudaLaunchKernel parity: stock={sx} monitor={mx}")
            else:
                for n in models:
                    for mode in MODES:
                        assert nkernels(mon[(n, mode)][1]) > 0, f"{n}/{mode}: no monitor kernels"

            serial = {f"{n}|{mode}": {name: {repr(ck): cnt for ck, cnt in cfgs.items()}
                                      for name, cfgs in mon[(n, mode)][1].items()}
                      for n in models for mode in MODES}
            print("RESULT", json.dumps(serial))
            print("OK")
            """
        )
        # Run parity once with HES off and once with it on (the HES child self-skips
        # via SKIP_HES when the platform doesn't support hardware events). Each child
        # prints its monitor metadata after RESULT; the HES-on monitor must match the
        # HES-off monitor (which already matched stock above).
        import json

        captured = {}
        for hes in (False, True):
            env = dict(os.environ)
            if hes:
                env["PARITY_HES"] = "1"
            p = subprocess.run(
                [sys.executable, "-c", script],
                text=True,
                capture_output=True,
                timeout=300,
                env=env,
            )
            self.assertEqual(
                p.returncode,
                0,
                f"subprocess failed (hes={hes}):\nstdout={p.stdout}\nstderr={p.stderr}",
            )
            if hes and "SKIP_HES" in p.stdout:
                continue
            self.assertIn("OK", p.stdout)
            line = next(
                (ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None
            )
            self.assertIsNotNone(line, f"no RESULT line (hes={hes}):\n{p.stdout}")
            captured[hes] = json.loads(line[len("RESULT ") :])
        if True in captured:
            self.assertEqual(
                captured[False],
                captured[True],
                "monitor kernel metadata differs with HES enabled vs disabled",
            )

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipUnless(
        SM100OrLater,
        "cuda_driver (cuLaunchKernelEx) activity is only emitted by the cuBLAS "
        "kernel-launch path on sm_100+; pre-sm_100 archs route launches through "
        "the runtime API and never surface a DRIVER record",
    )
    def test_cupti_monitor_observed_kinds_present(self):
        # Every activity kind the ProfilerObserver subscribes to must surface in the
        # exported chrome trace. One workload exercises kernels, H2D/D2H memcpy, memset,
        # runtime + driver API, CUPTI overhead, record_function annotations (external
        # correlation), and -- with enable_cuda_sync_events -- CUDA synchronization +
        # event records (cuda_sync, with the wait_on join the trace validator checks).
        import subprocess

        script = textwrap.dedent(
            """
            import gzip, json, os, tempfile, collections
            import torch, torch.nn as nn
            from torch.profiler import profile, ProfilerActivity, record_function
            from torch._C._profiler import _ExperimentalConfig
            from torch.profiler._trace_validator import validate_trace

            mlp = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(),
                                nn.Linear(4096, 1024), nn.GELU(),
                                nn.Linear(1024, 1024)).cuda().eval()

            def workload():
                x = torch.randn(64, 1024).cuda()          # H2D memcpy
                with record_function("region"):           # external correlation
                    y = mlp(x)                            # kernels/runtime/driver/memset
                y.cpu()                                    # D2H memcpy
                e = torch.cuda.Event(); e.record(); e.synchronize()
                torch.cuda.current_stream().wait_event(e)
                torch.cuda.synchronize()

            def categories(sync_on):
                cb = ('{"backend":"cupti_monitor","enable_cuda_sync_events":true}'
                      if sync_on else '{"backend":"cupti_monitor"}')
                cfg = _ExperimentalConfig(custom_profiler_config=cb)
                f = tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False)
                with torch.no_grad(), profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        experimental_config=cfg) as prof:
                    for _ in range(3):
                        workload()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(f.name)
                gz = f.name + ".gz"
                path = gz if os.path.exists(gz) else f.name
                ev = (json.load(gzip.open(gz)) if os.path.exists(gz)
                      else json.load(open(f.name)))["traceEvents"]
                cats = collections.Counter(e.get("cat") for e in ev if e.get("ph") == "X")
                ok, viol = validate_trace(path)
                os.remove(path)
                return cats, ok, [str(v) for v in viol]

            cats, ok, viol = categories(True)
            # CONCURRENT_KERNEL/MEMCPY/MEMSET/RUNTIME/DRIVER/OVERHEAD,
            # EXTERNAL_CORRELATION (-> user_annotation), SYNCHRONIZATION + CUDA_EVENT
            # (-> cuda_sync) must all be represented.
            expected = ["kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime",
                        "cuda_driver", "overhead", "user_annotation",
                        "gpu_user_annotation", "cuda_sync"]
            missing = [c for c in expected if not cats.get(c)]
            assert not missing, f"missing categories {missing}; got {dict(cats)}"
            assert ok, f"trace validator failed: {viol}"

            from torch.profiler._cupti.cupti_python import pylibcupti
            pylibcupti().finalize()
            off, _, _ = categories(False)
            assert not off.get("cuda_sync"), f"cuda_sync without opt-in: {dict(off)}"
            print("OK", dict(cats))
            """
        )
        p = subprocess.run(
            [sys.executable, "-c", script], text=True, capture_output=True, timeout=300
        )
        self.assertEqual(
            p.returncode, 0, f"subprocess failed:\nstdout={p.stdout}\nstderr={p.stderr}"
        )
        self.assertIn("OK", p.stdout)


class TestCuptiMonitorNative(TestCase):
    """The monitor's native buffer-pool / v2-record-layout callbacks driven directly
    via ctypes -- pure C++, no CUDA/cupti-python."""

    @skipIfTorchDynamo("native ctypes/CUPTI probe; nothing to compile")
    def test_cupti_monitor_buffer_pool_reuse(self):
        # The CUPTI monitor's buffer pool is pure C++ (no CUDA/cupti-python), so
        # drive its native buffer-requested / buffer-completed callbacks directly
        # via ctypes to verify returned buffers are recycled rather than
        # reallocated. The callbacks match cuptiActivityRegisterCallbacks_v2: the
        # request takes a trailing (ignored) info pointer, and the completion takes
        # the buffer + a complete-info pointer (no CUcontext/streamId -- those are
        # selectable record fields -- so completed buffers report ctx/stream of 0).
        import ctypes

        pyprof = torch._C._profiler
        pyprof._cupti_monitor.reset_buffers()
        self.addCleanup(pyprof._cupti_monitor.reset_buffers)
        buffer_size = 64 * 1024
        pyprof._cupti_monitor.configure_buffers(buffer_size)

        request_t = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        )
        complete_t = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        request = request_t(pyprof._cupti_monitor.buffer_request_callback_address())
        complete = complete_t(pyprof._cupti_monitor.buffer_complete_callback_address())

        def do_request():
            buf = ctypes.c_void_p()
            size = ctypes.c_size_t()
            max_records = ctypes.c_size_t()
            request(
                ctypes.byref(buf),
                ctypes.byref(size),
                ctypes.byref(max_records),
                None,  # CUpti_BufferCallbackRequestInfo*
            )
            return buf.value, size.value

        def do_complete(ptr):
            complete(ctypes.c_void_p(ptr), buffer_size, 4096, None)

        # First request has an empty free list, so it allocates.
        ptr_a, size_a = do_request()
        self.assertEqual(size_a, buffer_size)
        self.assertEqual(pyprof._cupti_monitor.allocated_buffers(), 1)

        # Complete it, drain it, and return it to the pool.
        do_complete(ptr_a)
        self.assertEqual(pyprof._cupti_monitor.pending_buffers(), 1)
        item = pyprof._cupti_monitor.get_completed()
        # (ptr, valid_size, ctx, stream, layouts): ctx/stream 0 (not delivered to the
        # completion callback) and layouts empty (driven with a null complete_info).
        self.assertEqual(item, (ptr_a, 4096, 0, 0, []))
        self.assertEqual(pyprof._cupti_monitor.pending_buffers(), 0)
        pyprof._cupti_monitor.return_buffer(ptr_a)

        # The next request reuses the freed buffer: same pointer, no new alloc.
        ptr_b, _ = do_request()
        self.assertEqual(ptr_b, ptr_a)
        self.assertEqual(pyprof._cupti_monitor.allocated_buffers(), 1)

        # A second concurrently-outstanding buffer forces a fresh allocation.
        ptr_c, _ = do_request()
        self.assertNotEqual(ptr_c, ptr_b)
        self.assertEqual(pyprof._cupti_monitor.allocated_buffers(), 2)

    def test_cupti_monitor_not_imported_without_active_session(self):
        # The optional CUPTI monitor import chain (observers.profiler -> monitor ->
        # cupti.cupti) must NOT be pulled in just by using record_function -- only an
        # active cupti_monitor profile imports it. Otherwise a process whose cupti-python
        # is too old for the monitor's symbols logs an import warning on every record
        # region. Checked in a fresh subprocess so other tests' imports don't pollute
        # sys.modules; needs no cupti-python.
        script = textwrap.dedent(
            """
            import sys
            import torch

            with torch.autograd.profiler.record_function("r"):
                pass
            leaked = sorted(m for m in sys.modules if m.startswith("torch.profiler._cupti"))
            assert not leaked, f"cupti chain imported without an active session: {leaked}"
            assert torch.autograd.profiler._active_cupti_profiler_observer is None
            print("OK")
            """
        )
        out = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.STDOUT
        )
        self.assertIn(b"OK", out)

    @skipIfTorchDynamo("native ctypes/CUPTI probe; nothing to compile")
    def test_cupti_monitor_v2_record_layout_capture(self):
        # The v2 complete callback parses the CUPTI user-defined record layout
        # (pBufferCompleteInfo->ppRecordLayouts, valid only during the callback) and
        # attaches it to the completed buffer, so the decode thread parses records
        # against each buffer's own layout. Build the CUPTI >= 13.3 complete-info /
        # record-layout structs with ctypes and drive the native v2 callbacks
        # directly (no CUDA/cupti-python); this also pins the C++ ABI mirror.
        import ctypes

        pyprof = torch._C._profiler
        pyprof._cupti_monitor.reset_buffers()
        self.addCleanup(pyprof._cupti_monitor.reset_buffers)
        pyprof._cupti_monitor.configure_buffers(64 * 1024)

        class FieldEntry(ctypes.Structure):
            _fields_ = [
                ("structSize", ctypes.c_size_t),
                ("fieldId", ctypes.c_int),
                ("offset", ctypes.c_size_t),
                ("size", ctypes.c_size_t),
                ("alignment", ctypes.c_size_t),
            ]

        class RecordLayout(ctypes.Structure):
            _fields_ = [
                ("structSize", ctypes.c_size_t),
                ("pEntries", ctypes.POINTER(FieldEntry)),
                ("numFields", ctypes.c_size_t),
                ("recordSize", ctypes.c_size_t),
            ]

        class CompleteInfo(ctypes.Structure):
            _fields_ = [
                ("structSize", ctypes.c_size_t),
                ("threadId", ctypes.c_uint64),
                ("ppRecordLayouts", ctypes.POINTER(ctypes.POINTER(RecordLayout))),
                ("numRecordLayouts", ctypes.c_size_t),
            ]

        # One activity kind (9) with two selected fields; the first must be the
        # *_FIELD_KIND id (0). ppRecordLayouts is indexed by kind, null elsewhere.
        entries = (FieldEntry * 2)(
            FieldEntry(ctypes.sizeof(FieldEntry), 0, 0, 4, 4),
            FieldEntry(ctypes.sizeof(FieldEntry), 5, 8, 8, 8),
        )
        layout = RecordLayout(
            ctypes.sizeof(RecordLayout),
            ctypes.cast(entries, ctypes.POINTER(FieldEntry)),
            2,
            16,
        )
        n_kinds = 10
        layouts_arr = (ctypes.POINTER(RecordLayout) * n_kinds)()
        layouts_arr[9] = ctypes.pointer(layout)
        info = CompleteInfo(
            ctypes.sizeof(CompleteInfo),
            1234,
            ctypes.cast(layouts_arr, ctypes.POINTER(ctypes.POINTER(RecordLayout))),
            n_kinds,
        )

        request_t = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        )
        complete_t = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        request = request_t(pyprof._cupti_monitor.buffer_request_callback_address())
        complete = complete_t(pyprof._cupti_monitor.buffer_complete_callback_address())

        buf = ctypes.c_void_p()
        size = ctypes.c_size_t()
        max_records = ctypes.c_size_t()
        request(ctypes.byref(buf), ctypes.byref(size), ctypes.byref(max_records), None)
        complete(
            ctypes.c_void_p(buf.value),
            16,
            16,
            ctypes.cast(ctypes.pointer(info), ctypes.c_void_p),
        )
        # The completed buffer carries CUPTI's parsed layout as its 5th field: the
        # per-kind (kind, record_size, [(field_id, offset, size), ...]) list (here
        # kind 9). No epoch / shared state -- the layout travels with the buffer.
        item = pyprof._cupti_monitor.get_completed()
        self.assertEqual(item[4], [(9, 16, [(0, 0, 4), (5, 8, 8)])])
        pyprof._cupti_monitor.return_buffer(item[0])

        # A second buffer with a different selection carries its own layout -- each
        # buffer decodes against the layout it was completed with.
        entries_b = (FieldEntry * 1)(FieldEntry(ctypes.sizeof(FieldEntry), 0, 0, 4, 4))
        layout_b = RecordLayout(
            ctypes.sizeof(RecordLayout),
            ctypes.cast(entries_b, ctypes.POINTER(FieldEntry)),
            1,
            8,
        )
        layouts_arr_b = (ctypes.POINTER(RecordLayout) * 4)()
        layouts_arr_b[3] = ctypes.pointer(layout_b)
        info_b = CompleteInfo(
            ctypes.sizeof(CompleteInfo),
            1234,
            ctypes.cast(layouts_arr_b, ctypes.POINTER(ctypes.POINTER(RecordLayout))),
            4,
        )
        request(ctypes.byref(buf), ctypes.byref(size), ctypes.byref(max_records), None)
        complete(
            ctypes.c_void_p(buf.value),
            8,
            8,
            ctypes.cast(ctypes.pointer(info_b), ctypes.c_void_p),
        )
        item_b = pyprof._cupti_monitor.get_completed()
        self.assertEqual(item_b[4], [(3, 8, [(0, 0, 4)])])
        pyprof._cupti_monitor.return_buffer(item_b[0])


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiClock(TestCase):
    """Clock-alignment math for the cupti_monitor -- record timestamps -> unix-epoch ns on
    kineto's axis. Drives the production _SynchronizedClock directly (its calibrate() reads
    the native clock through an injected callable), so no CUDA / live CUPTI session is needed;
    a lambda returning CLOCK_REALTIME stands in for cuptiGetTimestamp."""

    @staticmethod
    def _realtime():
        return time.clock_gettime_ns(time.CLOCK_REALTIME)

    def test_realtime_fallback_is_identity(self):
        # cuptiGetTimestamp == CLOCK_REALTIME == kineto's unix base: records are already unix
        # ns, so conversion is identity (offset 0) and 0 stays 0.
        import numpy as np

        from torch.profiler._cupti.monitor import _SynchronizedClock

        clock = _SynchronizedClock()
        clock.calibrate(callback_active=False, native_now=self._realtime)
        v = self._realtime() + 12_345
        self.assertEqual(clock.convert_unix(v), v)
        col = np.array([self._realtime(), 0, self._realtime() + 50], dtype=np.int64)
        np.testing.assert_array_equal(clock.convert_unix_array(col), col)

    def test_callback_scales_approx_ticks_to_unix(self):
        # Timestamp-callback path: records are approx ticks. calibrate() recovers the
        # converter's slope from two synthetic evaluations and applies it as a linear map (so
        # it never calls the converter per record); pass calibrate a converter we hold and
        # verify convert reproduces it -- to integer-rounding noise -- across a range of ticks.
        import numpy as np

        from torch.profiler._cupti.monitor import _SynchronizedClock

        conv = torch._C._profiler._ApproximateClockToUnixTimeConverter()
        clock = _SynchronizedClock()
        clock.calibrate(callback_active=True, native_now=self._realtime, converter=conv)
        a0 = clock.approx_anchor_ns
        ticks = [a0, a0 + 1000, a0 + 10_000_000, a0 + 5_000_000_000]
        for t in ticks:
            self.assertLessEqual(abs(clock.convert_approx(t) - conv.to_unix_ns(t)), 2)
        col = np.array([*ticks, 0], dtype=np.int64)
        got = clock.convert_approx_array(col)
        self.assertEqual(int(got[-1]), 0)  # 0 sentinel preserved
        for i, t in enumerate(ticks):
            self.assertLessEqual(abs(int(got[i]) - conv.to_unix_ns(t)), 2)

    def test_unix_array_bypasses_callback_scaling(self):
        # PM-sampling frames stay on the unix (CLOCK_REALTIME) domain even under the timestamp
        # callback, which moves *records* to the approx domain. convert_unix_array applies the
        # native->unix offset (~identity here, since native == realtime == unix), so native ns
        # land back on ~themselves and 0 stays 0 -- whereas convert_approx_array (the record path
        # under the callback) would misread the same column as approx ticks and diverge wildly.
        import numpy as np

        from torch.profiler._cupti.monitor import _SynchronizedClock

        conv = torch._C._profiler._ApproximateClockToUnixTimeConverter()
        clock = _SynchronizedClock()
        clock.calibrate(callback_active=True, native_now=self._realtime, converter=conv)
        col = np.array(
            [self._realtime(), 0, self._realtime() + 1_000_000], dtype=np.int64
        )
        got = clock.convert_unix_array(col)
        self.assertEqual(int(got[1]), 0)  # 0 sentinel preserved
        for i in (0, 2):
            self.assertLessEqual(abs(int(got[i]) - int(col[i])), 100_000)
        self.assertFalse(np.array_equal(clock.convert_approx_array(col), got))


if __name__ == "__main__":
    run_tests()
