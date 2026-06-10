# Owner(s): ["oncall: profiler"]
"""Tests for the CUPTI activity monitor and its v2/user-defined-record codec.

These are the non-profiler-specific monitor tests: the ``records`` field schema +
``decode`` codec (pure, no CUDA), and collection through ``CuptiMonitor``
directly (CUDA). Tests that exercise the monitor *through* ``torch.profiler.profile``
(trace shape, op/kernel-name parity, record_shapes, multithread, ...) live in
``test_profiler.py``.
"""

import ctypes
import threading
import time
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


# The CUPTI monitor needs libcupti >= 13.3: it uses the v2 user-defined-record API
# (>= 13.2) AND decodes against pBufferCompleteInfo->ppRecordLayouts (CUPTI's own
# per-kind record layout), which 13.2 leaves null. So a single >= 13.3 gate covers
# the whole monitor (it implies v2).
TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiRecords(TestCase):
    """Pure tests of the field schema + columnar codec -- no CUDA."""

    def test_field_catalog(self):
        # A Field carries (id, string); int(field) is its id. The per-kind catalogs
        # define the supported field ids; FIELD_REGISTRY and STRING_FIELDS derive
        # from them. No byte sizes -- those come from CUPTI's captured layout.
        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.records import FIELD_REGISTRY, Kernel, STRING_FIELDS

        self.assertEqual(int(Kernel.START), 7)
        self.assertEqual(Kernel.NAME.id, 24)
        self.assertTrue(Kernel.NAME.string)
        self.assertFalse(Kernel.START.string)
        kernel = int(ActivityKind.CONCURRENT_KERNEL)
        self.assertEqual(
            FIELD_REGISTRY[kernel], frozenset({0, 7, 8, 10, 11, 12, 22, 24, 31, 33})
        )
        self.assertEqual(STRING_FIELDS[kernel], frozenset({24}))

    def test_decode_columns(self):
        # records.decode demuxes a whole buffer (possibly interleaving
        # kinds) into {kind: {field_id: column}} against CUPTI's captured layout list
        # [(kind, record_size, [(field_id, offset, size), ...])] -- every field in
        # the layout. Three strategies: single kind (stride), uniform size (stride +
        # dispatch by KIND), variable size (per-record walk); plus a bounds guard
        # dropping a trailing record that overruns valid_size, and const char* fields
        # dereferenced in place. Synthetic layouts + buffers, no CUDA.
        import numpy as np

        from torch.profiler._cupti.cupti_python import ActivityKind
        from torch.profiler._cupti.monitor import CuptiMonitorBuffer

        kernel = int(ActivityKind.CONCURRENT_KERNEL)
        memcpy = int(ActivityKind.MEMCPY)

        def decode(addr, valid_size, record_layouts):
            # Decode a synthetic buffer. _returned=True so the RAII __del__ doesn't
            # hand this (non-pool) pointer back to the native buffer pool.
            buf = CuptiMonitorBuffer((addr, valid_size, 0, 0, record_layouts))
            buf._returned = True
            return buf.decode()

        def build(records, layouts):
            # records: [(kind, {field_id: int | bytes})]; a bytes value is written as
            # a real C string whose pointer is packed. layouts: {kind: (rsz, [(fid,
            # off, sz), ...])}. Returns (keepalive, addr, n).
            keep: list = []
            blob = bytearray()
            for kind, vals in records:
                rsz, fields = layouts[kind]
                rec = bytearray(rsz)
                values = {0: kind, **vals}
                for fid, off, sz in fields:
                    if fid not in values:
                        continue
                    v = values[fid]
                    if isinstance(v, bytes):
                        s = ctypes.create_string_buffer(v)
                        keep.append(s)
                        v = ctypes.addressof(s)
                    rec[off : off + sz] = int(v).to_bytes(sz, "little")
                blob += rec
            buf = (ctypes.c_uint8 * max(len(blob), 1)).from_buffer_copy(bytes(blob))
            keep.append(buf)
            return keep, ctypes.addressof(buf), len(blob)

        def as_list(layouts):
            return [(k, rsz, fields) for k, (rsz, fields) in layouts.items()]

        # --- single kind: homogeneous stride ---
        ker = {kernel: (24, [(0, 0, 4), (7, 8, 8), (8, 16, 8)])}
        keep, addr, n = build(
            [(kernel, {7: 100, 8: 150}), (kernel, {7: 200, 8: 275})], ker
        )
        out = decode(addr, n, as_list(ker))
        np.testing.assert_array_equal(out[kernel][7], [100, 200])
        np.testing.assert_array_equal(out[kernel][8], [150, 275])
        # Columns are independent copies of the buffer: scribbling over the source
        # bytes must not change already-decoded columns. This is what lets the worker
        # return the buffer to the pool immediately after decode().
        ctypes.memset(addr, 0, n)
        np.testing.assert_array_equal(out[kernel][7], [100, 200])
        np.testing.assert_array_equal(out[kernel][8], [150, 275])

        # --- uniform size: stride + dispatch by KIND (both kinds 24B) ---
        uni = {
            kernel: (24, [(0, 0, 4), (7, 8, 8), (8, 16, 8)]),
            memcpy: (24, [(0, 0, 4), (5, 8, 8), (6, 16, 8)]),
        }
        keepu, addru, nu = build(
            [
                (kernel, {7: 10, 8: 11}),
                (memcpy, {6: 20, 5: 999}),
                (kernel, {7: 30, 8: 33}),
            ],
            uni,
        )
        outu = decode(addru, nu, as_list(uni))
        np.testing.assert_array_equal(outu[kernel][7], [10, 30])
        np.testing.assert_array_equal(outu[memcpy][6], [20])
        np.testing.assert_array_equal(outu[memcpy][5], [999])

        # --- variable size + bounds guard: per-record walk ---
        # kernel (24B) and memcpy (16B) differ -> walk. A third kernel record's KIND
        # header fits in valid_size but its body runs past it -> dropped.
        var = {
            kernel: (24, [(0, 0, 4), (7, 8, 8)]),
            memcpy: (16, [(0, 0, 4), (6, 8, 8)]),
        }
        keepv, addrv, _ = build(
            [(kernel, {7: 1}), (memcpy, {6: 3}), (kernel, {7: 4})], var
        )
        valid = 24 + 16 + 4  # third kernel: KIND header only, body cut off
        outv = decode(addrv, valid, as_list(var))
        np.testing.assert_array_equal(outv[kernel][7], [1])  # truncated 3rd dropped
        np.testing.assert_array_equal(outv[memcpy][6], [3])

        # --- const char* string field dereferenced in place (NAME id 24) ---
        ks = {kernel: (40, [(0, 0, 4), (7, 8, 8), (24, 32, 8)])}
        keeps, addrs, ns = build([(kernel, {7: 7, 24: b"my_kernel"})], ks)
        outs = decode(addrs, ns, as_list(ks))
        self.assertEqual(list(outs[kernel][24]), ["my_kernel"])
        np.testing.assert_array_equal(outs[kernel][7], [7])

    def test_monitor_normalize_activities(self):
        # A registration request resolves to (kinds, per-kind field selection): a
        # bare kind iterable means "all fields"; a field map selects fields, with
        # "all"/None expanding; *_FIELD_KIND (0) is always included.
        from torch.profiler._cupti.cupti_python import ActivityKind
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

    def test_monitor_external_correlation_not_started(self):
        # External-correlation push/pop are no-ops until the monitor is started (no
        # subscriber yet), returning None rather than touching CUPTI's global stack.
        from torch.profiler._cupti.monitor import CuptiMonitor

        m = CuptiMonitor()
        self.assertFalse(m._started)
        self.assertIsNone(m.push_external_correlation_id())
        self.assertIsNone(m.pop_external_correlation_id())


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiMonitorCUDA(TestCase):
    """Collection through CuptiMonitor directly (not via torch.profiler.profile)."""

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_fence_enables_sync_transiently(self):
        # flush(sync=True) fences at a SYNCHRONIZATION sync point, enabled only for
        # the fence (even when no observer requested it) and disabled again after.
        from torch.profiler._cupti.cupti_python import ActivityKind
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
        monitor.flush(forced=True, sync=True)
        self.assertLess(time.time() - start, 2.0)
        self.assertNotIn(sync, monitor._enabled)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_v2_columnar_collection(self):
        # End-to-end columnar collection: the monitor turns on a per-activity field
        # selection, decodes each buffer against CUPTI's captured layout, and hands
        # the observer the columns for its selection.
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
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

        monitor.flush(forced=True, sync=True)
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
    def test_multiple_observers(self):
        # The monitor is the multiplexer: it enables the union of fields on its one
        # subscriber, then hands each observer only the columns it selected. Two
        # observers on the same kind with disjoint selections each see only their own
        # slice (plus KIND id 0) and the same set of records.
        from torch.profiler._cupti.cupti_python import ActivityKind, CuptiError
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
        monitor.flush(forced=True, sync=True)

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


if __name__ == "__main__":
    run_tests()
