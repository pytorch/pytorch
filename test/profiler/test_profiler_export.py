# Owner(s): ["oncall: profiler"]

import gzip
import json
import os
import sys
import tempfile
from unittest.mock import patch

from torch._utils import try_import
from torch.autograd.profiler import profile as _profile
from torch.profiler import _chrome_trace_export, ProfilerActivity
from torch.profiler.profiler import _KinetoProfile
from torch.testing._internal.common_profiler import forbid_profiler_capture
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryFileName,
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


if __name__ == "__main__":
    run_tests()
