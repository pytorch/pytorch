# Owner(s): ["oncall: profiler"]

import gzip
import json
import os
import sys
import tempfile
import unittest
from functools import cache
from unittest.mock import patch

import torch
from torch._utils import try_import
from torch.autograd.profiler import profile as _profile
from torch.profiler import (
    _chrome_trace_export,
    kineto_available,
    profile,
    ProfilerActivity,
)
from torch.profiler.profiler import _KinetoProfile
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_profiler import (
    forbid_profiler_capture,
    get_profiler_activities,
    initialize_kineto_with_accelerator,
    skipIfProfilerUnsupported,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryFileName,
    TEST_WITH_ROCM,
    TestCase,
)


class _StubActivity:
    """Minimal stand-in for ITraceActivity: the accessors the exporter calls."""

    def __init__(self, cat, name, rid, ts=100, metadata_json="", did=0):
        self._cat, self._name, self._rid = cat, name, rid
        self._ts, self._metadata_json, self._did = ts, metadata_json, did

    def type(self):
        return self._cat

    def name(self):
        return self._name

    def device_id(self):
        return self._did

    def resource_id(self):
        return self._rid

    def timestamp(self):
        return self._ts

    def duration(self):
        return 10

    def correlation_id(self):
        return 0

    def linked_correlation_id(self):
        return 0

    def linked_activity(self):
        return None

    def metadata_json(self):
        return self._metadata_json

    def flow_id(self):
        return 0

    def flow_start(self):
        return False

    def flow_type(self):
        return 2


class _StubResults:
    def __init__(self, activities):
        self._activities = activities
        self.saved = False

    def trace_activities(self):
        return self._activities

    def save(self, path):
        if self.saved:
            raise RuntimeError("Trace is already saved.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"traceEvents": []}, f)
        self.saved = True


@instantiate_parametrized_tests
class TestProfilerExport(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.enterContext(forbid_profiler_capture())
        # Synthetic Kineto results exercise the same routes on CPU-only builds.
        self.enterContext(
            patch("torch.autograd.profiler.kineto_available", return_value=True)
        )
        self.enterContext(patch("torch.autograd._add_metadata_json"))
        self.enterContext(
            patch.object(_chrome_trace_export, "_device_properties", return_value=[])
        )
        self.enterContext(
            patch.object(_chrome_trace_export, "_cuda_versions", return_value={})
        )

    def _make_profiler(self):
        activities = [_StubActivity("cpu_op", "aten::add", rid=1)]
        prof = _KinetoProfile(activities=[ProfilerActivity.CPU])
        prof.profiler = _profile(use_kineto=True)
        prof.profiler.kineto_results = _StubResults(activities)
        return prof

    def _load_trace(self, path, opener=open):
        def reject_duplicates(pairs):
            keys = [key for key, _ in pairs]
            self.assertEqual(len(keys), len(set(keys)), f"duplicate keys in {keys}")
            return dict(pairs)

        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=reject_duplicates)

    def _export(self, activities, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            _chrome_trace_export.export_chrome_trace(
                _StubResults(activities), path, **kwargs
            )
            return self._load_trace(path)

    @parametrize("use_python_export", [False, True])
    @parametrize(
        "suffix",
        [".json", ".json.gz", ".json.zst", "", ".unknown", ".zst.bak", ".gz.bak"],
    )
    def test_export(self, use_python_export, suffix):
        if suffix.endswith(".zst"):
            zstd = try_import("pyzstd") or try_import("zstandard")
            if zstd is None:
                self.skipTest("requires zstandard or pyzstd")
            opener = zstd.open
        elif suffix.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        value = "café 中"
        prof = self._make_profiler()
        prof.add_metadata_json(
            "compression_test", json.dumps(value, ensure_ascii=False)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace" + suffix)
            prof.export_chrome_trace(path, use_python_export=use_python_export)
            trace = self._load_trace(path, opener)
            if use_python_export:
                self.assertEqual(trace["compression_test"], value)
                names = [e["name"] for e in trace["traceEvents"]]
                self.assertIn("aten::add", names)
            self.assertEqual(prof.profiler.kineto_results.saved, not use_python_export)

    @parametrize("suffix", [".json.gz", ".json.zst"])
    def test_export_keeps_kineto_sibling(self, suffix):
        if suffix.endswith(".zst") and not (
            try_import("pyzstd") or try_import("zstandard")
        ):
            self.skipTest("requires zstandard or pyzstd")
        prof = self._make_profiler()
        kineto_export = prof.profiler.export_chrome_trace

        def export_with_sibling(path, *args, **kwargs):
            # Some Kineto builds write extra files next to the JSON.
            kineto_export(path, *args, **kwargs)
            with open(path.removesuffix(".json") + ".extra", "wb") as f:
                f.write(b"extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace" + suffix)
            with patch.object(
                prof.profiler, "export_chrome_trace", export_with_sibling
            ):
                prof.export_chrome_trace(path)
            sibling = os.path.join(tmpdir, "trace.extra")
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(sibling))
            with open(sibling, "rb") as f:
                self.assertEqual(f.read(), b"extra")
            # The uncompressed JSON stays in the temporary directory.
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "trace.json")))

    @parametrize("use_python_export", [False, True])
    def test_missing_zstd(self, use_python_export):
        prof = self._make_profiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json.zst")
            with patch.dict(sys.modules, {"zstandard": None, "pyzstd": None}):
                message = "requires zstandard or pyzstd.*pip install zstandard.*pip install pyzstd"
                with self.assertRaisesRegex(ImportError, message):
                    prof.export_chrome_trace(path, use_python_export=use_python_export)
                self.assertFalse(os.path.exists(path))
                self.assertFalse(prof.profiler.kineto_results.saved)
                # A missing dependency must not consume the native trace.
                gzip_path = os.path.join(tmpdir, "trace.json.gz")
                prof.export_chrome_trace(gzip_path, use_python_export=use_python_export)
                self.assertIn("traceEvents", self._load_trace(gzip_path, gzip.open))
                self.assertEqual(
                    prof.profiler.kineto_results.saved, not use_python_export
                )

    def test_export_chrome_trace_escapes_names(self):
        from torch.autograd.profiler_util import EventList, FunctionEvent

        tricky_name = 'aten::mm C:\\src\\"foo".cpp'
        evt = FunctionEvent(
            id=0,
            name=tricky_name,
            trace_name=tricky_name,
            thread=0,
            start_us=0,
            end_us=10,
        )
        event_list = EventList([evt])

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            trace = self._load_trace(fname)
        self.assertEqual(trace[0]["name"], tricky_name)

    @staticmethod
    def _metadata(graph_node_id):
        return f'"stream": 3, "graph node id": {graph_node_id}'

    @staticmethod
    def _kernels(trace):
        return [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]

    def test_annotation_fields_and_lane(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
            graph_lanes="all",
        )
        self.assertTrue(trace["cudaGraphInlineAnnotated"])
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["args"]["name"], "phase_a")
        # tid, args.stream and the annotation agree on the lane; the hardware stream
        # the work actually ran on is kept as original_stream.
        self.assertEqual(kernel["tid"], 61)
        self.assertEqual(kernel["args"]["stream"], 61)
        self.assertEqual(kernel["args"]["original_stream"], 3)
        # The lane gets a name of its own.
        names = [
            e
            for e in trace["traceEvents"]
            if e.get("name") == "thread_name" and e["tid"] == 61
        ]
        self.assertEqual(len(names), 1)

    def test_lane_named_by_annotation(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"Process Group Description": "DP", "stream": 61}]},
            graph_lanes="all",
        )
        [name_event] = [
            e
            for e in trace["traceEvents"]
            if e.get("name") == "thread_name" and e["tid"] == 61
        ]
        self.assertEqual(name_event["args"]["name"], "DP")

    def test_unannotated_graph_work_moves_to_default_stream(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={99: [{"name": "other"}]},
            graph_lanes="all",
        )
        # Nothing matched, so no inline flag -- but graphed work still collapses onto
        # the default lane instead of scattering over the replay's hardware streams.
        self.assertNotIn("cudaGraphInlineAnnotated", trace)
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 7)
        self.assertEqual(kernel["args"]["stream"], 7)
        self.assertEqual(kernel["args"]["original_stream"], 3)

    def test_non_graph_work_untouched(self):
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json='"stream": 3')],
            annotations={42: [{"name": "phase_a"}]},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertNotIn("original_stream", kernel["args"])

    def test_orphaned_user_annotation_slices_dropped(self):
        # CUPTI replicates gpu_user_annotation onto every stream a graph replays on.
        # A slice keeps its lane only if work stayed there: lane 3's kernel moved to 61.
        trace = self._export(
            [
                _StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42)),
                _StubActivity("gpu_user_annotation", "ann_moved", rid=3),
                _StubActivity("kernel", "eager", rid=9),
                _StubActivity("gpu_user_annotation", "ann_kept", rid=9),
            ],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
            graph_lanes="all",
        )
        kept = [
            e["name"]
            for e in trace["traceEvents"]
            if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(kept, ["ann_kept"])

    def test_invalid_graph_lanes_rejected(self):
        with self.assertRaisesRegex(ValueError, "graph_lanes"):
            self._export([], annotations={1: [{"name": "x"}]}, graph_lanes="lanes")

    def test_graph_lanes_all_without_annotations_rejected(self):
        # Standalone it would only collapse every graphed event onto default_stream.
        with self.assertRaisesRegex(ValueError, "needs cuda_graph_annotations"):
            self._export([], graph_lanes="all")

    def test_recorded_stream_reported_when_lanes_off(self):
        # graph_lanes="none" must not act on the recorded lane, but must not lose it
        # either: args.stream still names the stream the event actually ran on.
        trace = self._export(
            [_StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42))],
            annotations={42: [{"name": "phase_a", "stream": 61}]},
        )
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["args"]["name"], "phase_a")
        self.assertEqual(kernel["tid"], 3)
        self.assertEqual(kernel["args"]["stream"], 3)
        self.assertEqual(kernel["args"]["annotated_stream"], 61)
        self.assertNotIn("original_stream", kernel["args"])

    @parametrize("annotations", [None, {}])
    def test_without_annotations_nothing_changes(self, annotations):
        activities = [
            _StubActivity("kernel", "k", rid=3, metadata_json=self._metadata(42)),
            _StubActivity("gpu_user_annotation", "ann", rid=3),
        ]
        trace = self._export(activities, annotations=annotations)
        self.assertNotIn("cudaGraphInlineAnnotated", trace)
        [kernel] = self._kernels(trace)
        self.assertEqual(kernel["tid"], 3)
        self.assertNotIn("original_stream", kernel["args"])
        self.assertEqual(
            len(
                [
                    e
                    for e in trace["traceEvents"]
                    if e.get("cat") == "gpu_user_annotation"
                ]
            ),
            1,
        )


@cache
def _initialize_kineto_for_export():
    # Device test generation can call setUpClass during import; defer to setUp.
    return initialize_kineto_with_accelerator()


@unittest.skipIf(not kineto_available(), "Kineto is required")
@instantiate_parametrized_tests
@skipIfProfilerUnsupported
class TestProfilerTraceCompression(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        _initialize_kineto_for_export()

    @parametrize("use_python_export", [False, True])
    @parametrize(
        "suffix",
        [".json", ".json.gz", ".json.zst", ".unknown"],
    )
    def test_export(self, use_python_export, suffix):
        if suffix.endswith(".zst"):
            zstd = try_import("pyzstd") or try_import("zstandard")
            if zstd is None:
                self.skipTest("requires zstandard or pyzstd")
            opener = zstd.open
        elif suffix.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        value = "café 中"
        metadata = json.dumps(value, ensure_ascii=False)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            prof.add_metadata_json("compression_test", metadata)
            torch.ones(8) + 1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace" + suffix)
            prof.export_chrome_trace(path, use_python_export=use_python_export)
            with opener(path, "rt", encoding="utf-8") as f:
                trace = json.load(f)
            self.assertEqual(trace["compression_test"], value)
            self.assertTrue(trace["traceEvents"])

    def test_missing_zstd(self):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            torch.ones(8) + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json.zst")
            with patch.dict(sys.modules, {"zstandard": None, "pyzstd": None}):
                message = "requires zstandard or pyzstd.*pip install zstandard.*pip install pyzstd"
                with self.assertRaisesRegex(ImportError, message):
                    prof.export_chrome_trace(path)
                self.assertFalse(os.path.exists(path))
                # A missing dependency must not consume the native trace.
                gzip_path = os.path.join(tmpdir, "trace.json.gz")
                prof.export_chrome_trace(gzip_path)
                with gzip.open(gzip_path, "rt") as f:
                    self.assertIn("traceEvents", json.load(f))


@unittest.skipIf(not kineto_available(), "Kineto is required")
@skipIfProfilerUnsupported
class TestPythonChromeTraceExport(TestCase):
    """Verify that the Python streaming exporter produces traces equivalent
    to the C++ Kineto save() path."""

    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        _initialize_kineto_for_export()

    def test_python_export_matches_kineto(self, device):
        x = torch.randn(64, 64, device=device)
        with profile(activities=get_profiler_activities(device.split(":")[0])) as prof:
            for _ in range(20):
                torch.mm(x, x)
                torch.add(x, x)
            torch.accelerator.synchronize()

        with tempfile.TemporaryDirectory() as tmpdir:
            kineto_path = os.path.join(tmpdir, "kineto.json")
            prof.profiler.kineto_results.save(kineto_path)

            py_path = os.path.join(tmpdir, "python.json")
            prof.export_chrome_trace(py_path, use_python_export=True)

            with open(kineto_path) as f:
                kineto_trace = json.load(f)
            with open(py_path) as f:
                py_trace = json.load(f)

            gzip_path = os.path.join(tmpdir, "python.json.gz")
            prof.export_chrome_trace(gzip_path, use_python_export=True)
            with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
                gzip_trace = json.load(f)
            self.assertEqual(gzip_trace["traceEvents"], py_trace["traceEvents"])

        kineto_events = kineto_trace["traceEvents"]
        py_events = py_trace["traceEvents"]

        def _activity_events(events):
            return [
                e
                for e in events
                if e.get("ph") == "X" and e.get("cat") not in ("Trace",)
            ]

        kineto_acts = _activity_events(kineto_events)
        py_acts = _activity_events(py_events)
        self.assertGreater(len(kineto_acts), 0)
        self.assertEqual(
            len(kineto_acts),
            len(py_acts),
            lambda msg: f"{msg}\nActivity event count mismatch: kineto={len(kineto_acts)}, python={len(py_acts)}",
        )

        from collections import Counter

        kineto_names = Counter((e["cat"], e["name"]) for e in kineto_acts)
        py_names = Counter((e["cat"], e["name"]) for e in py_acts)
        self.assertEqual(
            kineto_names, py_names, "Event (cat, name) distribution differs"
        )

        def _strip_ev_idx(event):
            """Return a copy with the internal 'Ev Idx' arg removed."""
            e = dict(event)
            args = dict(e.get("args", {}))
            args.pop("Ev Idx", None)
            e["args"] = args
            return e

        kineto_sorted = sorted(
            [_strip_ev_idx(e) for e in kineto_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        py_sorted = sorted(
            [_strip_ev_idx(e) for e in py_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        # Compare every key of every activity event.
        for ke, pe in zip(kineto_sorted, py_sorted):
            self.assertEqual(
                ke,
                pe,
                lambda msg: f"{msg}\nEvent mismatch for {ke['cat']}::{ke['name']} at ts={ke['ts']}",
            )

        # Flow event parity (ph, id, cat).
        kineto_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in kineto_events
                if e.get("ph") in ("s", "f")
            ],
        )
        py_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in py_events
                if e.get("ph") in ("s", "f")
            ],
        )
        self.assertEqual(kineto_flow, py_flow, "Flow events differ")


instantiate_device_type_tests(
    TestPythonChromeTraceExport, globals(), except_for=("cpu",)
)


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@skipIfProfilerUnsupported
class TestPythonChromeTraceExportCudaOnly(TestCase):
    """Python streaming exporter checks keyed on CUDA-specific data (CUDA runtime/
    driver versions, CUDA graph annotations) that have no cross-backend equivalent."""

    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        _initialize_kineto_for_export()

    def _profile_workload(self):
        x = torch.randn(64, 64, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                torch.mm(x, x)
                torch.add(x, x)
            torch.cuda.synchronize()
        return prof

    # The envelope is NVIDIA-shaped on both sides, and on ROCm the two sides disagree:
    # kineto's ROCm backend writes hip_driver_version/hip_runtime_version instead of the
    # cuda_* keys, and its device-property set names two fields differently
    # (maxSharedMemoryPerMultiProcessor, regsPerBlock) from the cudaDeviceProp shape the
    # exporter reproduces. The event-stream comparisons above hold on ROCm; only the
    # envelope has no defined ROCm equivalence yet.
    @unittest.skipIf(TEST_WITH_ROCM, "Chrome-trace envelope is not validated on ROCm")
    def test_python_export_envelope_matches_kineto(self):
        """Top-level trace keys, which the profiler result does not expose and the
        exporter therefore recomputes: device properties and the CUDA versions kineto
        writes from logGpuVersions()."""
        prof = self._profile_workload()

        with tempfile.TemporaryDirectory() as tmpdir:
            kineto_path = os.path.join(tmpdir, "kineto.json")
            prof.profiler.kineto_results.save(kineto_path)
            py_path = os.path.join(tmpdir, "python.json")
            prof.export_chrome_trace(py_path, use_python_export=True)

            with open(kineto_path) as f:
                kineto_trace = json.load(f)
            with open(py_path) as f:
                py_trace = json.load(f)

        self.assertEqual(
            py_trace["cuda_runtime_version"], kineto_trace["cuda_runtime_version"]
        )
        # The Python exporter omits the driver version when the optional cuda-bindings
        # package is unavailable or cannot query its CUDA runtime.
        if "cuda_driver_version" in py_trace:
            self.assertEqual(
                py_trace["cuda_driver_version"], kineto_trace["cuda_driver_version"]
            )
        # regsPerBlock is the one field torch's device properties do not carry, so the
        # exporter omits it; everything else must match what kineto wrote.
        for py_device, kineto_device in zip(
            py_trace["deviceProperties"], kineto_trace["deviceProperties"]
        ):
            self.assertEqual(
                py_device,
                {k: v for k, v in kineto_device.items() if k != "regsPerBlock"},
            )

    def test_cuda_graph_annotations_reach_the_trace(self):
        """End to end: annotations recorded at capture are baked into a replay's
        kernel events by export_chrome_trace(cuda_graph_annotations=...)."""
        from torch.cuda._graph_annotations import _reset_kernel_annotations
        from torch.cuda.graph_annotations import (
            get_kernel_annotations,
            is_available,
            mark_kernels,
        )

        if not is_available():
            self.skipTest("CUDA graph annotations are unavailable")

        _reset_kernel_annotations()
        self.addCleanup(_reset_kernel_annotations)
        x = torch.randn(64, 64, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, enable_annotations=True):
            with mark_kernels({"name": "phase_a", "stream": 61}):
                _ = x + 1
        self.assertGreater(len(get_kernel_annotations()), 0)

        # The first profiled iterations can drop kernels while CUPTI warms up.
        for _ in range(3):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "trace.json")
                prof.export_chrome_trace(
                    path,
                    cuda_graph_annotations=get_kernel_annotations(),
                    graph_lanes="all",
                )
                with open(path) as f:
                    trace = json.load(f)
            annotated = [
                e
                for e in trace["traceEvents"]
                if e.get("args", {}).get("name") == "phase_a"
            ]
            if annotated:
                break

        self.assertTrue(trace.get("cudaGraphInlineAnnotated"))
        self.assertGreater(len(annotated), 0)
        for event in annotated:
            self.assertEqual(event["tid"], 61)
            self.assertEqual(event["args"]["stream"], 61)
            self.assertIn("original_stream", event["args"])


if __name__ == "__main__":
    run_tests()
