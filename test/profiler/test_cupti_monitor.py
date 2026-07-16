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


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiRecords(TestCase):
    """Pure monitor + metadata unit tests (no CUDA)."""

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

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_monitor_buffer_size_from_env(self):
        # The per-buffer pool size is user-configurable: an explicit buffer_size
        # arg wins, otherwise TORCH_CUPTI_MONITOR_BUFFER_SIZE is honored, else the
        # 4 MiB default. No CUDA -- this only reads the constructor config.
        import unittest.mock

        from torch.profiler._cupti.monitor import _DEFAULT_BUFFER_SIZE, CuptiMonitor

        self.assertEqual(CuptiMonitor().buffer_size, _DEFAULT_BUFFER_SIZE)
        with unittest.mock.patch.dict(
            "os.environ", {"TORCH_CUPTI_MONITOR_BUFFER_SIZE": "1048576"}
        ):
            self.assertEqual(CuptiMonitor().buffer_size, 1048576)
            # An explicit arg overrides the env var.
            self.assertEqual(CuptiMonitor(buffer_size=2048).buffer_size, 2048)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_monitor_use_approx_timestamps_arg(self):
        # The approx-clock timestamp callback is a constructor setting (no env var):
        # off by default, on when use_approx_timestamps=True. No CUDA, but constructing
        # CuptiMonitor loads libcupti (>= 13.3) via pylibcupti().
        from torch.profiler._cupti.monitor import CuptiMonitor

        self.assertFalse(CuptiMonitor()._timestamp_callback_enabled)
        self.assertTrue(
            CuptiMonitor(use_approx_timestamps=True)._timestamp_callback_enabled
        )

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


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiMonitorCUDA(TestCase):
    """Collection through CuptiMonitor directly (not via torch.profiler.profile)."""

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
    def test_approx_timestamp_callback_engages_under_udr(self):
        # use_approx_timestamps hands CUPTI a per-subscriber timestamp callback
        # (CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK) so records ride the profiler's approx
        # clock. Unlike the global cuptiActivityRegisterTimestampCallback (NOT_COMPATIBLE under
        # user-defined records), the subscriber attribute coexists with the UDR path. It only
        # re-times device records when the monitor subscribes before any CUDA context exists
        # (cuptiSubscribe otherwise latches device correlation on the pre-existing context), so
        # this runs in an inline-script child that never imports common_cuda (which would init
        # CUDA at import) and subscribes first. Assert the callback engaged and that device
        # records land on the approx clock (~1e14-1e15 ticks), not CLOCK_REALTIME ns (~1e18).
        script = textwrap.dedent(
            """
            import torch
            from cupti.cupti import ActivityKind
            from torch.profiler._cupti import monitor as mon
            from torch.profiler._cupti.records import Kernel

            assert not torch.cuda.is_initialized()
            kind = ActivityKind.CONCURRENT_KERNEL
            seen = []
            m = mon.CuptiMonitor(use_approx_timestamps=True)
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
    def test_approx_timestamp_callback_skipped_with_existing_context(self):
        # cuptiSubscribe latches device-record correlation against a pre-existing CUDA context,
        # so the (post-subscribe) per-subscriber callback can't re-time device records. When a
        # context already exists, the monitor must refuse to engage rather than silently drop
        # device records; collection still proceeds on the CLOCK_REALTIME pass-through.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.monitor import CuptiMonitor
        from torch.profiler._cupti.records import Kernel

        kind = ActivityKind.CONCURRENT_KERNEL
        want = {kind: {Kernel.START, Kernel.END}}
        torch.randn(8, device="cuda").sum().item()
        torch.cuda.synchronize()
        self.assertTrue(torch.cuda.is_initialized())

        lock = threading.Lock()
        columns: list = []
        monitor = CuptiMonitor(use_approx_timestamps=True)

        def on_columns(cols):
            if kind in cols:
                with lock:
                    columns.append(cols[kind])

        try:
            obs = monitor.register(want, on_columns)
        except CuptiError as e:
            self.skipTest(f"monitor could not subscribe: {e}")
        self.addCleanup(monitor.unregister, obs)
        self.assertFalse(monitor._timestamp_callback_active)

        x = torch.randn(256, 256, device="cuda")
        for _ in range(4):
            x = torch.relu(x @ x)
        x.sum().item()
        torch.cuda.synchronize()
        monitor.flush(sync=True)
        monitor.unregister(obs)

        self.assertGreater(sum(len(c[int(Kernel.START)]) for c in columns), 0)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_singleton_flush_accessible(self):
        # A user can reach the process-wide monitor singleton through the public
        # accessors and flush it: instance() constructs/returns it, get_monitor()
        # hands back that same object, and flush(sync=True) on the singleton
        # delivers everything collected up to the call.
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

        mon = cupti_monitor.instance()
        self.assertIs(cupti_monitor.get_monitor(), mon)
        # Drop the singleton after the observer is torn down so the next instance()
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
        cupti_monitor.get_monitor().flush(sync=True)

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

        m = CuptiMonitor(buffer_size=1024, flush_period_s=0.02)
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

        monitor = _cupti_monitor.instance()
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

            monitor = _cupti_monitor.instance()
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
