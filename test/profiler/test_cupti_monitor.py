# Owner(s): ["oncall: profiler"]

import gzip
import json
import os
import sys
import threading
import unittest

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity, record_function
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.utils._import_utils import _check_module_exists


# cupti-python is pip-installable on ROCm hosts too, but CUPTI itself is a no-op
# there, so gate the monitor tests off ROCm as well.
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestCuptiMonitorCUDA(TestCase):
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

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    def test_cupti_monitor_collection_raw_dump_smoke(self):
        from torch.profiler._cupti import monitor as _cupti_monitor

        with TemporaryDirectoryName() as out_dir:
            self.assertIsNone(_cupti_monitor.get_monitor())
            monitor = _cupti_monitor.start_collection(out_dir)
            self.assertIs(monitor, _cupti_monitor.get_monitor())

            x = torch.randn(64, 64, device="cuda")
            y = torch.relu(x + 1)
            y.sum().item()
            torch.cuda.synchronize()

            stats = _cupti_monitor.stop_collection()
            self.assertIsNotNone(stats)
            self.assertIsNone(_cupti_monitor.get_monitor())
            # The native C++ pool must actually have been exercised: catches a
            # silent regression to a no-op (e.g. broken callback registration or
            # symbol export) that would still produce passing file-existence
            # checks if the worker never saw a buffer.
            self.assertGreater(stats["buffers_allocated"], 0)
            self.assertGreater(stats["buffers_completed"], 0)
            self.assertEqual(stats["buffers_pending"], 0)
            self.assertTrue(
                os.path.exists(os.path.join(out_dir, _cupti_monitor._META_FILE))
            )
            self.assertTrue(
                os.path.exists(os.path.join(out_dir, _cupti_monitor._RAW_BUFFER_FILE))
            )
            self.assertGreater(
                os.path.getsize(os.path.join(out_dir, _cupti_monitor._META_FILE)), 0
            )
            self.assertGreater(
                os.path.getsize(os.path.join(out_dir, _cupti_monitor._RAW_BUFFER_FILE)),
                0,
            )

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    def test_cupti_monitor_collection_repeated_lifecycle(self):
        from torch.profiler._cupti import monitor as _cupti_monitor

        for _ in range(2):
            with TemporaryDirectoryName() as out_dir:
                self.assertIsNone(_cupti_monitor.get_monitor())
                _cupti_monitor.start_collection(out_dir)

                x = torch.randn(32, 32, device="cuda")
                y = torch.sigmoid(x)
                y.sum().item()
                torch.cuda.synchronize()

                stats = _cupti_monitor.stop_collection()
                self.assertIsNotNone(stats)
                self.assertIsNone(_cupti_monitor.get_monitor())

                self.assertTrue(
                    os.path.exists(os.path.join(out_dir, _cupti_monitor._META_FILE))
                )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(out_dir, _cupti_monitor._RAW_BUFFER_FILE)
                    )
                )
                self.assertGreater(
                    os.path.getsize(
                        os.path.join(out_dir, _cupti_monitor._RAW_BUFFER_FILE)
                    ),
                    0,
                )

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
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

            opener = gzip.open if trace_path.endswith(".gz") else open
            with opener(trace_path, "rt") as f:
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

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
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

    @unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
    def test_cupti_monitor_matches_stock_op_and_kernel_names(self):
        def trace_summary(use_monitor):
            cfg = _ExperimentalConfig(
                custom_profiler_config='{"backend":"cupti_monitor"}'
                if use_monitor
                else ""
            )
            with TemporaryFileName(mode="w+") as trace_path:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    experimental_config=cfg,
                ) as prof:
                    a = torch.randn(128, 128, device="cuda")
                    b = torch.randn(128, 128, device="cuda")
                    (a @ b).relu().sum()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(trace_path)
                with open(trace_path) as f:
                    events = json.load(f)["traceEvents"]
            aten_ops = {
                e["name"]
                for e in events
                if e.get("cat") == "cpu_op" and e.get("name", "").startswith("aten::")
            }
            n_kernels = sum(
                1 for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"
            )
            return aten_ops, n_kernels

        stock_ops, stock_kernels = trace_summary(use_monitor=False)
        monitor_ops, monitor_kernels = trace_summary(use_monitor=True)
        self.assertGreater(stock_kernels, 0)
        self.assertGreater(monitor_kernels, 0)
        self.assertEqual(monitor_ops, stock_ops)


class TestCuptiMonitorBufferPool(TestCase):
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


if __name__ == "__main__":
    run_tests()
