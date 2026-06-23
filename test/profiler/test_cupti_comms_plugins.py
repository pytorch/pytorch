# Owner(s): ["oncall: profiler"]
"""Tests for the in-core CUPTI comms serializer plugin: FlightRecorderPlugin (the OSS
FlightRecorder schema fr_trace reads). The ``CommsObserver`` producer,
``CommMonitorHook``, and ``HangDetectorPlugin`` are in ``test_cupti_comms.py``;
service-specific serializers (torchcomms clog, ncclx comm_dump) live with their
consumers, not in core.
"""

import json
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.utils._import_utils import _check_module_exists


TEST_CUDA = torch.cuda.is_available()
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


def _cupti_version() -> int:
    if not TEST_CUPTI_PYTHON:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti

        return pylibcupti().get_version()
    except Exception:
        return 0


TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiCommsPlugins(TestCase):
    def test_flight_recorder_plugin(self):
        # Serializes CommRecords to the OSS FlightRecorder schema (fr_trace input):
        # an in-flight collective is a "scheduled" entry, then "completed" with timing
        # once recorded. Backend-supplied metadata (process_group, sizes, ranks) flows
        # through under the OSS key names. Host-side.
        from torch.profiler._cupti.comms import FlightRecorderPlugin
        from torch.profiler._cupti.observers.comms import CommRecord

        plugin = FlightRecorderPlugin(comm_lib_version="nccl-2.99")

        meta = {
            "func": "AllReduce",
            "datatype": "ncclBfloat16",
            "count": 1024,
            "seq": 7,
            # backend-supplied schema fields (via the metadata store):
            "process_group": ["0", "default"],
            "process_group_ranks": [0, 1],
            "input_sizes": [[1024]],
            "output_sizes": [[1024]],
        }
        # Issued: appears as a scheduled entry (the hang-suspect state).
        plugin.on_schedule(42, meta)
        sched = plugin.dump()["entries"]
        self.assertEqual(len(sched), 1)
        self.assertEqual(sched[0]["state"], "scheduled")
        self.assertEqual(sched[0]["profiling_name"], "nccl:all_reduce")
        self.assertIsNone(sched[0]["duration_ms"])

        # Completed: same entry flips to completed with device timing.
        plugin.on_end(
            CommRecord(
                coll_id=42,
                name="ncclDevKernel_AllReduce",
                start_ns=1000,
                end_ns=3000,
                graph_node_id=0,
                metadata=meta,
            )
        )
        dump = plugin.dump()

        self.assertEqual(dump["version"], "2.10")
        self.assertEqual(dump["comm_lib_version"], "nccl-2.99")
        self.assertEqual(
            dump["pg_config"]["0"], {"name": "0", "desc": "default", "ranks": "[0, 1]"}
        )
        self.assertEqual(dump["pg_status"]["0"]["last_completed_collective"], 7)
        self.assertEqual(len(dump["entries"]), 1)
        e = dump["entries"][0]
        self.assertEqual(e["state"], "completed")
        self.assertEqual(e["collective_seq_id"], 7)
        self.assertEqual(e["input_sizes"], [[1024]])
        self.assertEqual(e["duration_ms"], 2000 / 1e6)
        self.assertEqual(e["process_group"], ["0", "default"])
        # Every field the fr_trace analyzer reads is present and JSON-serializable.
        for key in (
            "record_id",
            "pg_id",
            "process_group",
            "collective_seq_id",
            "p2p_seq_id",
            "op_id",
            "profiling_name",
            "time_created_ns",
            "input_sizes",
            "output_sizes",
            "state",
            "time_discovered_started_ns",
            "time_discovered_completed_ns",
            "retired",
            "is_p2p",
        ):
            self.assertIn(key, e)
        json.loads(plugin.dump_json())  # round-trips

    def test_flight_recorder_plugin_clock_converter(self):
        # clock_converter maps CUPTI native ns -> unix-epoch ns on the three timestamp
        # fields (for cross-rank correlation / tlparse), while duration_ms (a delta)
        # stays native. None keeps native ns (default). Host-side.
        from torch.profiler._cupti.comms import FlightRecorderPlugin
        from torch.profiler._cupti.observers.comms import CommRecord

        meta = {"func": "AllReduce", "count": 1024, "seq": 0}
        record = CommRecord(
            coll_id=42,
            name="ncclDevKernel_AllReduce",
            start_ns=1000,
            end_ns=3000,
            graph_node_id=0,
            metadata=meta,
        )

        plugin = FlightRecorderPlugin(clock_converter=lambda ns: ns + 1_000_000)
        plugin.on_end(record)
        e = plugin.dump()["entries"][0]
        self.assertEqual(e["time_created_ns"], 1000 + 1_000_000)
        self.assertEqual(e["time_discovered_started_ns"], 1000 + 1_000_000)
        self.assertEqual(e["time_discovered_completed_ns"], 3000 + 1_000_000)
        self.assertEqual(e["duration_ms"], 2000 / 1e6)  # delta unchanged

        # Default (no converter): timestamps stay native ns.
        plain = FlightRecorderPlugin()
        plain.on_end(record)
        e = plain.dump()["entries"][0]
        self.assertEqual(e["time_discovered_started_ns"], 1000)
        self.assertEqual(e["time_discovered_completed_ns"], 3000)
        self.assertEqual(e["duration_ms"], 2000 / 1e6)

    def test_flight_recorder_fr_trace_detects_hang(self):
        # End-to-end: feed two ranks' FlightRecorderPlugin dumps (rank 0 completed the
        # allreduce, rank 1 still scheduled = hung) to the real fr_trace analyzer and
        # confirm it parses our schema and names the culprit. Proves the CUPTI monitor
        # can drive the existing PyTorch analyzer unchanged. Host-side, no CUDA.
        import argparse

        try:
            from torch.distributed.flight_recorder.components.builder import build_db
        except ImportError:
            self.skipTest("torch.distributed.flight_recorder unavailable")
        from torch.profiler._cupti.comms import FlightRecorderPlugin
        from torch.profiler._cupti.observers.comms import CommRecord

        meta = {
            "func": "AllReduce",
            "datatype": "ncclBfloat16",
            "count": 1024,
            "seq": 0,
            "process_group": ["0", "default"],
            "process_group_ranks": [0, 1],
            "input_sizes": [[1024]],
            "output_sizes": [[1024]],
        }

        def dump_for(rank, completed):
            p = FlightRecorderPlugin()
            p.on_schedule(42, meta)  # scheduled
            if completed:
                p.on_end(
                    CommRecord(
                        coll_id=42,
                        name="ncclDevKernel_AllReduce",
                        start_ns=1000,
                        end_ns=3000,
                        graph_node_id=0,
                        metadata=meta,
                    )
                )
            d = p.dump()
            d["rank"] = rank
            return d

        details = {"0": dump_for(0, True), "1": dump_for(1, False)}
        args = argparse.Namespace(
            verbose=False,
            just_print_entries=False,
            allow_incomplete_ranks=False,
            mismatch_cap=10,
        )
        with self.assertLogs("Flight Recorder", level="INFO") as cm:
            build_db(details, args, version="2.10")
        output = "\n".join(cm.output)
        self.assertIn("COLLECTIVE_STATE_MISMATCH", output)
        self.assertIn("Culprit rank 1", output)

    def test_flight_recorder_write_dump_read_by_fr_trace(self):
        # write_dump's on-disk format/convention: write two ranks' per-rank pickles to
        # a dir, then read them back with fr_trace's own read_dir + build_db and
        # confirm the culprit is flagged -- proving the dump transport (the on-timeout
        # output path), not just the in-memory dict. Host-side.
        import argparse
        import os
        import tempfile

        try:
            from torch.distributed.flight_recorder.components.builder import build_db
            from torch.distributed.flight_recorder.components.loader import read_dir
        except ImportError:
            self.skipTest("torch.distributed.flight_recorder unavailable")
        from torch.profiler._cupti.comms import FlightRecorderPlugin
        from torch.profiler._cupti.observers.comms import CommRecord

        meta = {
            "func": "AllReduce",
            "datatype": "ncclBfloat16",
            "count": 1024,
            "seq": 0,
            "process_group": ["0", "default"],
            "process_group_ranks": [0, 1],
            "input_sizes": [[1024]],
            "output_sizes": [[1024]],
        }

        def plugin_for(completed):
            p = FlightRecorderPlugin()
            p.on_schedule(42, meta)
            if completed:
                p.on_end(
                    CommRecord(
                        coll_id=42,
                        name="ncclDevKernel_AllReduce",
                        start_ns=1000,
                        end_ns=3000,
                        graph_node_id=0,
                        metadata=meta,
                    )
                )
            return p

        with tempfile.TemporaryDirectory() as d:
            p0 = plugin_for(True).write_dump(rank=0, path=os.path.join(d, "trace_0"))
            plugin_for(False).write_dump(rank=1, path=os.path.join(d, "trace_1"))
            self.assertTrue(os.path.exists(p0))
            args = argparse.Namespace(
                trace_dir=d,
                prefix="trace_",
                verbose=False,
                just_print_entries=False,
                allow_incomplete_ranks=False,
                mismatch_cap=10,
            )
            details, version = read_dir(args)
            self.assertEqual(version, "2.10")
            with self.assertLogs("Flight Recorder", level="INFO") as cm:
                build_db(details, args, version=version)
        output = "\n".join(cm.output)
        self.assertIn("COLLECTIVE_STATE_MISMATCH", output)
        self.assertIn("Culprit rank 1", output)

    def test_flight_recorder_dump_on_hang(self):
        # The on-timeout wiring: a HangDetectorPlugin whose on_timeout calls
        # FlightRecorderPlugin.write_dump produces the per-rank pickle when the observer
        # reports a collective stuck via its quiescence stall heartbeat (on_progress).
        # The observer -> on_progress link is covered in test_cupti_comms.py; here we
        # drive on_progress directly. Fires once per stuck collective (dedup).
        import os
        import pickle
        import tempfile

        from torch.profiler._cupti.comms import FlightRecorderPlugin, HangDetectorPlugin

        fr = FlightRecorderPlugin()
        # One issued-but-not-completed collective for the dump to capture.
        fr.on_schedule(42, {"func": "AllReduce", "process_group": ["0", "default"]})

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace_0")
            hang = HangDetectorPlugin(on_timeout=lambda *_: fr.write_dump(path=path))
            self.assertFalse(os.path.exists(path))  # nothing reported stuck yet

            hang.on_progress({42: {"func": "AllReduce"}})  # stall heartbeat -> dump
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                dump = pickle.load(f)
            self.assertEqual(dump["version"], "2.10")
            self.assertEqual(len(dump["entries"]), 1)
            self.assertEqual(dump["entries"][0]["state"], "scheduled")

            # Re-reporting the same stuck collective does not re-fire (dedup).
            os.remove(path)
            hang.on_progress({42: {"func": "AllReduce"}})
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    run_tests()
