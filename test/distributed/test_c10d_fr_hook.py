# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the FlightRecorder hook
# (torch._C._distributed_c10d.FlightRecorderHook): FlightRecorder recording
# driven by the ProcessGroup pre/post collective hooks rather than native
# backend integration, so it works for any backend routed through c10d ops.
# Modeled on torchcomms' hooks/fr FlightRecorderTest; parameterized over
# backends like test_c10d_fault_tolerance.py.

import json
import os
import pickle
import sys
import tempfile
import time
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch._C._distributed_c10d import FlightRecorderHook
from torch.distributed.distributed_c10d import _world
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


FR_HOOK_BACKENDS = [
    ("gloo", "cpu"),
    ("nccl2", "cuda"),
]


class AbstractFlightRecorderHookTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.rank}")
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        # Note: gloo also records natively into the same recorder; the hook
        # deliberately writes a fixed "c10d:" prefix instead of the backend
        # name, so keying entries by it both tells the two apart and keeps the
        # assertions backend-agnostic.
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        # Spawned children do not re-run setUp, so they pick the dump prefix
        # out of the inherited environment rather than off self.
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = os.path.join(
            self.tempdir.name, "trace_"
        )
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("TORCH_FR_DUMP_TEMP_FILE", None)
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _dump_file_name(self):
        return os.environ["TORCH_FR_DUMP_TEMP_FILE"] + str(self.rank)

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    def _all_entries(self):
        trace = json.loads(torch._C._distributed_c10d._dump_fr_trace_json())
        return trace.get("entries", [])

    def _hook_entries(self):
        return [
            e for e in self._all_entries() if e["profiling_name"].startswith("c10d:")
        ]

    @property
    def _records_natively(self):
        # ProcessGroupGloo writes into the same FlightRecorder<c10::Event> the
        # hook uses, so groups holding it are skipped by the auto-attach.
        return self.backend_name == "gloo"

    def _fr_hook(self, pg):
        # Groups whose backends have no native recording already got a hook at
        # creation time; reuse it rather than attaching a second one, which
        # would record every collective twice.
        hook = _world.pg_flight_recorder_hooks.get(pg)
        return hook if hook is not None else FlightRecorderHook.attach(pg)

    def test_records_and_retires_collectives(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        names = [e["profiling_name"] for e in entries]
        self.assertIn("c10d:all_reduce", names)
        self.assertIn("c10d:broadcast", names)
        self.assertIn("c10d:barrier", names)
        # Post-hooks fire right after the op is issued, so every recorded op
        # must be retired. On a device backend the entry also carries the
        # hook's start/end events, so its state comes from querying them at
        # retire time ("scheduled" until the start event is observed
        # complete); on CPU there are no events and retired is the only
        # completion signal, as with Gloo's native FR recording.
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
        hook.remove()

    def test_entry_state_from_device_events(self):
        self._init_pg()
        # A fresh subgroup initializes its communicators lazily, so the first
        # collective on it spends long enough in the backend for the start
        # event recorded by the pre-hook to be complete by the time the
        # post-hook retires the entry.
        sub_pg = dist.new_group(list(range(self.world_size)))
        hook = self._fr_hook(sub_pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        for _ in range(3):
            dist.all_reduce(t, group=sub_pg)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        self.assertEqual(len(entries), 3)
        discovered = [e for e in entries if e["time_discovered_started_ns"] > 0]
        if self.device_type == "cpu":
            # Null events: nothing to query, so no discovery at all.
            self.assertEqual(discovered, [])
            for e in entries:
                self.assertEqual(e["state"], "scheduled", msg=str(e))
        else:
            self.assertTrue(discovered, msg=str(entries))
            for e in discovered:
                self.assertIn(e["state"], ("started", "completed"), msg=str(e))
        hook.remove()

    def test_duration_ms_populated(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        self.assertEqual(len(entries), 3)
        # The post-hook retires at issue time, so the end event has usually not
        # signalled yet and duration_ms comes from the host clock instead (an
        # upper bound, not kernel time). On CPU there are no events at all and
        # it is always the host clock. Either way every retired entry has one.
        for e in entries:
            self.assertIn("duration_ms", e, msg=str(e))
            self.assertGreater(e["duration_ms"], 0.0, msg=str(e))
            self.assertLess(e["duration_ms"], 60000.0, msg=str(e))
        hook.remove()

    def test_records_tensor_metadata(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, 8, device=self.device)
        dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        allreduce = [e for e in entries if e["profiling_name"] == "c10d:all_reduce"]
        self.assertEqual(len(allreduce), 1)
        self.assertEqual(allreduce[0]["input_sizes"], [[4, 8]])
        self.assertEqual(allreduce[0]["input_dtypes"], ["Float"])
        hook.remove()

    def test_p2p_and_collective_seq_ids(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        send_t = torch.ones(4, device=self.device)
        recv_t = torch.empty(4, device=self.device)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(send_t, peer)
            dist.recv(recv_t, peer)
        else:
            dist.recv(recv_t, peer)
            dist.send(send_t, peer)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        p2p = [e for e in entries if e["is_p2p"]]
        coll = [e for e in entries if e["profiling_name"] == "c10d:all_reduce"]
        self.assertEqual(len(p2p), 2)
        self.assertEqual(len(coll), 1)
        # P2P ops advance p2p_seq_id only; collectives advance
        # collective_seq_id only.
        self.assertEqual(sorted(e["p2p_seq_id"] for e in p2p), [1, 2])
        self.assertEqual(coll[0]["collective_seq_id"], 1)
        # The peer goes into the name the way the analyzer parses it, with
        # group-local ranks.
        self.assertEqual(
            sorted(e["profiling_name"] for e in p2p),
            sorted(
                [
                    f"c10d:send {self.rank}->{peer}",
                    f"c10d:recv {self.rank}<-{peer}",
                ]
            ),
        )
        hook.remove()

    def test_entries_parse_with_trace_analyzer(self):
        # The dumped entries must be readable by torchfrtrace; parsing them
        # with its Op class is what catches a profiling_name that the analyzer
        # rejects (unknown backend prefix, unknown op name, missing p2p peer).
        from torch.distributed.flight_recorder.components.types import Op

        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.all_gather([torch.empty_like(t) for _ in range(self.world_size)], t)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(t, peer)
        else:
            dist.recv(t, peer)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        self.assertGreaterEqual(len(entries), 4)
        for e in entries:
            pg_name = e["process_group"][0]
            memberships = {pg_name: set(range(self.world_size))}
            op = Op(e, memberships, pg_name)
            if op.type in ("send", "recv"):
                self.assertEqual(op.src if op.type == "send" else op.dst, self.rank)
        hook.remove()

    def test_dump_to_file(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        torch._C._distributed_c10d._dump_fr_trace_file(self.rank)
        with open(self._dump_file_name(), "rb") as f:
            dump = pickle.load(f)
        self.assertIn("version", dump)
        self.assertIn("entries", dump)
        self.assertIn("pg_config", dump)
        names = [e["profiling_name"] for e in dump["entries"]]
        self.assertIn("c10d:all_reduce", names)
        hook.remove()

    def test_control_plane_dump_file(self):
        from torch._C._distributed_c10d import _get_handler, _Request, _Response

        class Request(_Request):
            def body(self):
                return b""

            def params(self):
                return {}

        class Response(_Response):
            status = None
            content = None

            def set_content(self, content, content_type):
                self.content = content

            def set_status(self, status):
                self.status = status

        pg = self._init_pg()
        # attach() is what tells the recorder which rank it is running on,
        # which is how the handler names the file.
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        res = Response()
        _get_handler("fr_dump_file")(Request(), res)
        self.assertEqual(res.status, 200)

        # The handler dumps on a worker thread, so poll until the file is
        # there and complete -- it appears empty before the write lands.
        path = self._dump_file_name()
        dump = None
        deadline = time.time() + 60
        while dump is None and time.time() < deadline:
            try:
                with open(path, "rb") as f:
                    dump = pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError):
                time.sleep(0.1)
        self.assertIsNotNone(dump, msg=f"no dump written to {path}")
        self.assertIn("version", dump)
        self.assertIn("entries", dump)
        self.assertIn("pg_config", dump)
        hook.remove()

    def test_remove_stops_recording(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        count_attached = len(self._hook_entries())
        self.assertGreater(count_attached, 0)

        hook.remove()
        dist.all_reduce(t)
        self.assertEqual(len(self._hook_entries()), count_attached)

    def test_multiple_collectives_entry_order(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        for _ in range(5):
            dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        seqs = [
            e["collective_seq_id"]
            for e in entries
            if e["profiling_name"] == "c10d:all_reduce"
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 5)
        hook.remove()

    def test_auto_attach_records_without_explicit_attach(self):
        # TORCH_FR_BUFFER_SIZE is set in setUp, so creating the group is all it
        # takes for a backend with no native recording (nccl2) to be traced.
        pg = self._init_pg()
        attached = pg in _world.pg_flight_recorder_hooks
        self.assertEqual(attached, not self._records_natively)

        before = len(self._hook_entries())
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        names = [e["profiling_name"] for e in self._hook_entries()[before:]]
        if self._records_natively:
            self.assertEqual(names, [])
        else:
            self.assertEqual(names, ["c10d:all_reduce", "c10d:broadcast"])

    def test_no_double_recording(self):
        # A collective must produce exactly one entry: the hook's for a backend
        # that has no native recording, the backend's own otherwise. Gloo writes
        # to the same recorder under the same (group_name, group_desc) key, so
        # an auto-attached hook there would show up as a second entry.
        self._init_pg()
        before = len(self._all_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        names = [e["profiling_name"] for e in self._all_entries()[before:]]
        expected = "gloo:all_reduce" if self._records_natively else "c10d:all_reduce"
        self.assertEqual(names, [expected])

    def test_reset_fr_trace(self):
        # No explicit hook: each backend records a collective exactly once
        # (gloo natively, nccl2 through the auto-attached hook), so the entry
        # counts below are exact.
        self._init_pg()

        t = torch.ones(4, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self.assertGreaterEqual(len(self._all_entries()), 3)

        torch._C._distributed_c10d._reset_fr_trace()
        # Soft delete: the ring buffer keeps the entries but bumps the reset
        # epoch, so a dump filters out everything recorded before the reset.
        self.assertEqual(self._all_entries(), [])

        for _ in range(2):
            dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        entries = self._all_entries()
        self.assertEqual(len(entries), 2)
        # Record ids restart at 0 for the new epoch.
        self.assertEqual([e["record_id"] for e in entries], [0, 1])

    def test_attach_without_abort_hook_support(self):
        # Abort hooks are what trigger the dump on a collective failure, but
        # they are optional: gloo has none, and Backend's default
        # registerAbortHook throws rather than no-opping. attach() must ask
        # first and keep working -- only the dump-on-failure trigger is lost,
        # recording is unaffected.
        pg = self._init_pg()
        backend = pg._get_backend(self.device)
        supported = self.backend_name == "nccl2"
        self.assertEqual(pg.supports_abort_hooks, supported)
        self.assertEqual(backend.supports_abort_hooks, supported)
        if not supported:
            with self.assertRaisesRegex(
                RuntimeError, "does not support registerAbortHook"
            ):
                backend.register_abort_hook(0, lambda: None)

        before = len(self._hook_entries())
        hook = self._fr_hook(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self.assertEqual(len(self._hook_entries()), before + 1)
        # remove() must not throw either: unregisterAbortHook throws on a
        # backend that has none, so it may only be called if attach()
        # registered one.
        hook.remove()

    def test_buffer_size_zero_attaches_nothing(self):
        # The gate is read when the group is created, so flip it before init.
        saved = os.environ["TORCH_FR_BUFFER_SIZE"]
        os.environ["TORCH_FR_BUFFER_SIZE"] = "0"
        try:
            pg = self._init_pg()
            sub_pg = dist.new_group(list(range(self.world_size)))
        finally:
            os.environ["TORCH_FR_BUFFER_SIZE"] = saved
        self.assertNotIn(pg, _world.pg_flight_recorder_hooks)
        self.assertNotIn(sub_pg, _world.pg_flight_recorder_hooks)

        before = len(self._hook_entries())
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.all_reduce(t, group=sub_pg)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self.assertEqual(self._hook_entries()[before:], [])


def _make_fr_hook_test_class(backend_name, device_type):
    class FlightRecorderHookTest(AbstractFlightRecorderHookTest, MultiProcessTestCase):
        pass

    FlightRecorderHookTest.backend_name = backend_name
    FlightRecorderHookTest.device_type = device_type
    FlightRecorderHookTest.__name__ = (
        f"{backend_name.capitalize()}FlightRecorderHookTest"
    )
    FlightRecorderHookTest.__qualname__ = FlightRecorderHookTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(FlightRecorderHookTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "FR hook CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in FR_HOOK_BACKENDS:
    globals()[f"{backend_name.capitalize()}FlightRecorderHookTest"] = (
        _make_fr_hook_test_class(backend_name, device_type)
    )


if __name__ == "__main__":
    run_tests()
