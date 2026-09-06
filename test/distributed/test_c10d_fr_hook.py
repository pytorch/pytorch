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
from torch.testing._internal.distributed.fake_pg import FakeStore


# CPU coverage of the hook comes from "fake" rather than gloo: gloo has its own
# FlightRecorder integration, so the hook deliberately skips its ops and
# recording nothing is all a gloo group can show. What is left to cover on CPU
# is the null-start-event path, and "fake" exercises it through the same c10d
# ops as any other backend.
FR_HOOK_BACKENDS = [
    ("fake", "cpu"),
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
        os.environ["LOCAL_RANK"] = str(self.rank)
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        if self._communicates:
            store = dist.FileStore(self.file_name, self.world_size)
        else:
            store = FakeStore()
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    def _all_entries(self, backend=None):
        trace = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(
                backend=backend if backend is not None else self.backend_name
            )
        )
        return trace.get("entries", [])

    def _name(self, op):
        # The hook writes the backend's own name as the comm_lib field, so
        # profiling names are backend-specific.
        return f"{self.backend_name}:{op}"

    def _hook_entries(self):
        # Every backend under test here is hooked, and each hooked backend
        # records into a recorder instance of its own, so nothing else writes
        # to the one being read.
        return self._all_entries()

    @property
    def _auto_attached(self):
        # _maybe_attach_flight_recorder skips a group whose every device is
        # served by a backend that either records itself or never
        # communicates, so "fake" has to be attached by hand.
        return self.backend_name == "nccl2"

    @property
    def _communicates(self):
        return self.backend_name != "fake"

    @property
    def _push_completion(self):
        # Two tiers. A backend with completion hooks pushes real completion, so
        # an entry is retired when its collective actually finished and reads
        # "completed" with the backend's own duration. A backend without them
        # ("fake") is retired by the post-hook, at issue: honest but degraded,
        # since nothing will ever tell the hook the op finished.
        return self.backend_name == "nccl2"

    def _completed_state(self):
        return "completed" if self._push_completion else "scheduled"

    def _await_retired(self, count=0, timeout=60.0):
        # Completion arrives when the backend establishes it: on its watchdog,
        # which ticks about once a second, or on the next collective through the
        # same group. Nothing polls at dump time any more, so a dump taken the
        # instant after cuda.synchronize() can be one tick early. Returns as soon
        # as it can, which for the retire-at-issue tier is immediately.
        deadline = time.time() + timeout
        while True:
            entries = self._hook_entries()
            done = len(entries) >= count and all(e["retired"] for e in entries)
            if done or time.time() >= deadline:
                return entries
            time.sleep(0.05)

    def _fr_hook(self, pg):
        # Groups whose backends have no native recording already got a hook at
        # creation time; reuse it rather than attaching a second one, which
        # would record every collective twice. A hand-attached hook is given
        # the same global rank mapping _maybe_attach_flight_recorder passes,
        # which is the only source of one for a backend that does not fill in
        # Options::global_ranks_in_group.
        hook = _world.pg_flight_recorder_hooks.get(pg)
        if hook is not None:
            return hook
        return FlightRecorderHook.attach(pg, dist.get_process_group_ranks(pg))

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

        entries = self._await_retired(before + 3)[before:]
        names = [e["profiling_name"] for e in entries]
        self.assertIn(self._name("all_reduce"), names)
        self.assertIn(self._name("broadcast"), names)
        self.assertIn(self._name("barrier"), names)
        # Every entry is retired, on both tiers. Only a backend that pushes
        # completion can also say the collective finished.
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
            self.assertEqual(e["state"], self._completed_state(), msg=str(e))
        hook.remove()

    def test_entry_state_from_pushed_completion(self):
        # Tier one: the backend pushes completion from where its watchdog
        # establishes it, so "completed" is reachable without the hook polling
        # anything. See test_retires_at_issue_without_completion_hooks for the
        # other tier.
        if not self._push_completion:
            self.skipTest("backend pushes no completion")
        self._init_pg()
        sub_pg = dist.new_group(list(range(self.world_size)))
        hook = self._fr_hook(sub_pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        for _ in range(3):
            dist.all_reduce(t, group=sub_pg)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._await_retired(before + 3)[before:]
        self.assertEqual(len(entries), 3)
        # Discovery is when the backend told the hook, so it is always set for a
        # completed entry.
        for e in entries:
            self.assertEqual(e["state"], "completed", msg=str(e))
            self.assertGreater(e["time_discovered_completed_ns"], 0, msg=str(e))
            self.assertGreater(e["time_discovered_started_ns"], 0, msg=str(e))
        hook.remove()

    def test_retires_at_issue_without_completion_hooks(self):
        # Tier two: a backend with no completion hook. The post-hook has to
        # retire the entry itself, at issue, because nothing else ever will and
        # an entry nothing retires is indistinguishable from a hang. What it must
        # not do is claim a completion nobody observed, so the entry reads
        # "scheduled" and carries no duration.
        if self._push_completion:
            self.skipTest("backend pushes completion")
        pg = self._init_pg()
        self.assertFalse(pg.supports_completion_hooks)
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        self.assertEqual(len(entries), 3)
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
            self.assertEqual(e["state"], "scheduled", msg=str(e))
            self.assertEqual(e["time_discovered_completed_ns"], 0, msg=str(e))
            self.assertNotIn("duration_ms", e, msg=str(e))
        hook.remove()

    def test_duration_ms_absent_without_backend_timing(self):
        # duration_ms comes from the backend, which reports none unless it was
        # asked to time collectives. There is deliberately no host clock stand-in:
        # the report arrives on a watchdog tick, so a wall clock would measure how
        # late the push was, not the collective.
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._await_retired(before + 3)[before:]
        self.assertEqual(len(entries), 3)
        for e in entries:
            self.assertEqual(e["state"], self._completed_state(), msg=str(e))
            self.assertNotIn("duration_ms", e, msg=str(e))
        hook.remove()

    def test_duration_ms_from_backend_timing(self):
        # With timing on, duration_ms is the backend's own device measurement
        # of the collective, not a host-side upper bound.
        pg = self._init_pg()
        if self.backend_name != "nccl2":
            self.skipTest("backend cannot time collectives")
        # Before the first collective: works created earlier carry untimed
        # events, so their completion can report no duration.
        pg._enable_collectives_timing()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(1024, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        entries = self._await_retired(before + 3)[before:]
        self.assertEqual(len(entries), 3)
        for e in entries:
            self.assertIn("duration_ms", e, msg=str(e))
            self.assertGreater(e["duration_ms"], 0.0, msg=str(e))
            self.assertLess(e["duration_ms"], 60000.0, msg=str(e))
        hook.remove()

    def test_hung_collective_reads_not_completed(self):
        # The property the whole design exists for: a dump taken while a
        # collective is stuck must show that collective as not completed, and
        # must still show the ones before it as completed. Rank 1 joins late,
        # so rank 0's second all_reduce is genuinely in flight while it dumps.
        if not self._communicates or self.device_type != "cuda":
            self.skipTest("needs a backend whose collectives really block")
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        before = len(self._await_retired(1))

        if self.rank == 0:
            # Returns as soon as it is issued: a c10d wait() on CUDA only
            # orders the caller's stream after the collective.
            dist.all_reduce(t)
            # Long enough for several watchdog ticks, so this is the backend
            # having found nothing to report rather than not having looked.
            time.sleep(5)
            entries = self._hook_entries()
            hung = entries[before:]
            self.assertEqual(len(hung), 1, msg=str(hung))
            self.assertNotEqual(hung[0]["state"], "completed", msg=str(hung[0]))
            self.assertFalse(hung[0]["retired"], msg=str(hung[0]))
            self.assertEqual(hung[0]["time_discovered_completed_ns"], 0)
            # ... and the healthy ones before it are unaffected.
            self.assertEqual(entries[before - 1]["state"], "completed")
        else:
            time.sleep(15)
            dist.all_reduce(t)

        torch.cuda.synchronize()
        # Once it finishes, the backend pushes that too.
        self.assertEqual(self._await_retired()[-1]["state"], "completed")
        hook.remove()

    def test_op_names_distinguish_collective_variants(self):
        # The op name is what the analyzer keys its per-collective size rules
        # off, so every dispatcher op has to keep its own spelling. Folding
        # _allgather_base into "all_gather" made the list-form numel rule apply
        # to a flattened buffer, and folding alltoall_base into "all_to_all"
        # meant a rank doing one could match a peer doing the other.
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        n = self.world_size
        shard = torch.ones(8, device=self.device)
        flat = torch.ones(8 * n, device=self.device)
        dist.all_reduce(shard)
        dist.all_gather([torch.empty_like(shard) for _ in range(n)], shard)
        dist.all_gather_into_tensor(torch.empty_like(flat), shard)
        dist.reduce_scatter(
            torch.empty_like(shard), [torch.ones_like(shard) for _ in range(n)]
        )
        dist.reduce_scatter_tensor(torch.empty_like(shard), flat)
        dist.all_to_all(list(torch.empty_like(flat).chunk(n)), list(flat.chunk(n)))
        dist.all_to_all_single(torch.empty_like(shard), shard)
        with dist._coalescing_manager():
            for _ in range(2):
                dist.all_reduce(torch.ones(4, device=self.device))
        with dist._coalescing_manager():
            for _ in range(2):
                dist.all_gather_into_tensor(torch.empty_like(flat), shard)
        with dist._coalescing_manager():
            for _ in range(2):
                dist.reduce_scatter_tensor(torch.empty_like(shard), flat)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        self.assertEqual(
            [e["profiling_name"] for e in entries],
            [
                self._name(op)
                for op in (
                    "all_reduce",
                    "all_gather",
                    "_all_gather_base",
                    "reduce_scatter",
                    "_reduce_scatter_base",
                    "all_to_all",
                    "all_to_all_single",
                    "allreduce_coalesced",
                    "all_gather_into_tensor_coalesced",
                    "reduce_scatter_tensor_coalesced",
                )
            ],
        )
        # Every one of them has to be a name the analyzer accepts, or the
        # entry is dropped from the timeline entirely.
        from torch.distributed.flight_recorder.components.types import Op

        for e in entries:
            pg_name = e["process_group"][0]
            Op(e, {pg_name: set(range(n))}, pg_name)
        # allreduce_coalesced has no output tensors of its own either, so it
        # needs the same input mirroring as all_reduce.
        coalesced = entries[-3]
        self.assertEqual(coalesced["input_sizes"], coalesced["output_sizes"])
        hook.remove()

    def test_recv_any_source_records_unknown_peer(self):
        # recv_any_source has no peer until a message arrives. Writing -1 made
        # the analyzer index the group's rank list from the end and pin the
        # recv on the highest-ranked member.
        from torch.distributed.flight_recorder.components.types import Op

        if self.backend_name != "fake":
            # NCCL has no recvAnysource; only mpi, ucc and fake do.
            self.skipTest("backend does not support recv_any_source")
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        # Not dist.recv(t, src=None): it asks the returned Work who the sender
        # was, which the fake backend cannot answer.
        pg.recv_anysource([torch.empty(4, device=self.device)], 0).wait()

        entries = self._hook_entries()[before:]
        self.assertEqual(len(entries), 1, msg=str(entries))
        e = entries[0]
        self.assertEqual(e["profiling_name"], self._name(f"recv {self.rank}<-?"))
        pg_name = e["process_group"][0]
        op = Op(e, {pg_name: set(range(self.world_size))}, pg_name)
        self.assertIsNone(op.src)
        self.assertIsNone(op._src_g)
        self.assertEqual(op.dst, self.rank)
        hook.remove()

    def test_subgroup_publishes_real_global_ranks(self):
        # A subgroup's membership and this rank's dump file name both come from
        # the global rank mapping. Falling back to 0..size-1 when the backend
        # has none made every member of a subgroup claim a global rank that
        # belongs to someone else: the analyzer then saw a membership that
        # never existed, and -- since the file is <prefix><rank> and the loader
        # parses the rank back out of the name -- several ranks wrote the same
        # file and all but one post-mortem was lost.
        import ast

        import requests

        from torch._C._distributed_c10d import _WorkerServer

        last = self.world_size - 1
        if last == 0:
            self.skipTest("needs a subgroup that excludes rank 0")
        pg = self._init_pg()
        self._fr_hook(pg)
        sub = dist.new_group([last])
        if self.rank != last:
            return
        hook = self._fr_hook(sub)

        dist.all_reduce(torch.ones(4, device=self.device), group=sub)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        trace = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(backend=self.backend_name)
        )
        ranks = ast.literal_eval(trace["pg_config"][sub.group_name]["ranks"])
        self.assertEqual(ranks, [last])

        # The handler names the file from the rank the recorder was told, so
        # this is what catches setRank being overwritten with a group-local one.
        server = _WorkerServer("", 0)
        self.addCleanup(server.shutdown)
        res = requests.post(
            f"http://localhost:{server.port}/handler/fr_dump_file"
            f"?backend={self.backend_name}",
            timeout=60,
        )
        self.assertEqual(res.status_code, 200)
        path = self._dump_file_name()
        deadline = time.time() + 60
        while not os.path.exists(path) and time.time() < deadline:
            time.sleep(0.1)
        self.assertTrue(os.path.exists(path), msg=f"no dump written to {path}")
        self.assertFalse(os.path.exists(os.environ["TORCH_FR_DUMP_TEMP_FILE"] + "0"))
        hook.remove()

    def test_attach_without_rank_mapping_publishes_nothing(self):
        # Nothing is invented when neither the caller nor the backend can say
        # what this group's global ranks are: the collectives are still
        # recorded, but no membership is published for the group.
        if self.backend_name != "fake":
            # Backends that fill in Options::global_ranks_in_group cannot
            # reach this path.
            self.skipTest("backend supplies its own global rank mapping")
        last = self.world_size - 1
        pg = self._init_pg()
        self._fr_hook(pg)
        sub = dist.new_group([last])
        if self.rank != last:
            return
        hook = FlightRecorderHook.attach(sub)

        before = len(self._hook_entries())
        dist.all_reduce(torch.ones(4, device=self.device), group=sub)

        trace = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(backend=self.backend_name)
        )
        self.assertEqual(len(trace["entries"]) - before, 1)
        self.assertNotIn(sub.group_name, trace["pg_config"])
        hook.remove()

    def test_records_tensor_metadata(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, 8, device=self.device)
        dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        allreduce = [
            e for e in entries if e["profiling_name"] == self._name("all_reduce")
        ]
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
        coll = [e for e in entries if e["profiling_name"] == self._name("all_reduce")]
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
                    self._name(f"send {self.rank}->{peer}"),
                    self._name(f"recv {self.rank}<-{peer}"),
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

    def test_hook_trace_analyzes_without_mismatch(self):
        # The property the recording exists for: every rank's entries, fed to
        # torchfrtrace's own analysis, must produce no mismatch. The hook
        # records what the dispatcher hands it, which for these is not the
        # shape a native backend records -- all_reduce, reduce and broadcast
        # have no output tensor of their own, and the list forms have one
        # shard-shaped buffer per rank instead of one flattened buffer. Each
        # used to be reported as a cross-rank size mismatch, which then poisons
        # the process group so no later collective can match either.
        from torch.distributed.flight_recorder.components.builder import build_db
        from torch.distributed.flight_recorder.components.config_manager import (
            JobConfig,
        )

        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.reduce(t, dst=0)
        dist.broadcast(t, src=0)
        dist.all_gather([torch.empty_like(t) for _ in range(self.world_size)], t)
        dist.reduce_scatter(
            torch.empty_like(t), [torch.ones_like(t) for _ in range(self.world_size)]
        )
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        # Before the cross-rank comparison: an entry the backend has not reported
        # yet reads "scheduled" on one rank and "completed" on another, which the
        # analyzer is right to call a state mismatch.
        entries = self._await_retired(before + 5)[before:]
        trace = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(backend=self.backend_name)
        )
        collectives = (
            "all_reduce",
            "reduce",
            "broadcast",
            "all_gather",
            "reduce_scatter",
        )
        self.assertEqual(
            [e["profiling_name"] for e in entries],
            [self._name(c) for c in collectives],
        )
        shard = [8]
        self.assertEqual(
            [(e["input_sizes"], e["output_sizes"]) for e in entries],
            [
                ([shard], [shard]),
                ([shard], [shard]),
                ([shard], [shard]),
                ([shard], [shard] * self.world_size),
                ([shard] * self.world_size, [shard]),
            ],
        )

        for e in entries:
            self.assertEqual(e["state"], self._completed_state(), msg=str(e))
        local = {
            "entries": entries,
            "pg_config": trace["pg_config"],
            "version": trace["version"],
        }
        if self._communicates:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local)
        else:
            # The backend moves no data, so a gather would return nothing
            # usable; every rank issued the same collectives, which is what the
            # cross-rank comparison below needs.
            gathered = [local] * self.world_size
        details = {
            f"trace_{r}": {"host_name": f"host_rank{r}", "rank": r, **g}
            for r, g in enumerate(gathered)
        }
        db = build_db(details, JobConfig().parse_args([]), trace["version"])
        self.assertEqual(len(db.collectives), len(collectives))
        for c in db.collectives:
            self.assertTrue(
                c.pass_check, msg=f"{c.collective_name}: {c.type_of_mismatch}"
            )
        hook.remove()

    def test_dump_to_file(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        torch._C._distributed_c10d._dump_fr_trace_file(
            self.rank, backend=self.backend_name
        )
        with open(self._dump_file_name(), "rb") as f:
            dump = pickle.load(f)
        self.assertIn("version", dump)
        self.assertIn("entries", dump)
        self.assertIn("pg_config", dump)
        names = [e["profiling_name"] for e in dump["entries"]]
        self.assertIn(self._name("all_reduce"), names)
        hook.remove()

    def test_control_plane_dump_file(self):
        # Over the real control plane rather than _get_handler, because the
        # backend to dump arrives as a query parameter and Python cannot
        # implement Request::params().
        import requests

        from torch._C._distributed_c10d import _WorkerServer

        pg = self._init_pg()
        # attach() is what tells the recorder which rank it is running on,
        # which is how the handler names the file.
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        server = _WorkerServer("", 0)
        self.addCleanup(server.shutdown)
        res = requests.post(
            f"http://localhost:{server.port}/handler/fr_dump_file"
            f"?backend={self.backend_name}",
            timeout=60,
        )
        self.assertEqual(res.status_code, 200)

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
        names = [e["profiling_name"] for e in dump["entries"]]
        self.assertIn(self._name("all_reduce"), names)
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
            if e["profiling_name"] == self._name("all_reduce")
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 5)
        hook.remove()

    def test_auto_attach_records_without_explicit_attach(self):
        # TORCH_FR_BUFFER_SIZE is set in setUp, so creating the group is all it
        # takes for a backend that needs the hook (nccl2) to be traced.
        pg = self._init_pg()
        self.assertEqual(pg in _world.pg_flight_recorder_hooks, self._auto_attached)

        before = len(self._hook_entries())
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        names = [e["profiling_name"] for e in self._hook_entries()[before:]]
        if self._auto_attached:
            self.assertEqual(names, [self._name("all_reduce"), self._name("broadcast")])
        else:
            self.assertEqual(names, [])

    def test_records_only_into_its_own_instance(self):
        # A collective produces exactly one entry, and it lands in the hooked
        # backend's own recorder instance. Nothing reaches the default
        # instance, which is the one ProcessGroupGloo records into.
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._all_entries())
        before_default = len(self._all_entries(backend="gloo"))

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        names = [e["profiling_name"] for e in self._all_entries()[before:]]
        self.assertEqual(names, [self._name("all_reduce")])
        self.assertEqual(len(self._all_entries(backend="gloo")), before_default)
        hook.remove()

    def test_reset_fr_trace(self):
        pg = self._init_pg()
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        for _ in range(3):
            dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self.assertGreaterEqual(len(self._all_entries()), 3)

        torch._C._distributed_c10d._reset_fr_trace(backend=self.backend_name)
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
        hook.remove()

    def test_attach_without_abort_hook_support(self):
        # Abort hooks are what trigger the dump on a collective failure, but
        # they are optional: most backends have none, and Backend's default
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

    def test_attach_without_completion_hook_support(self):
        # Completion hooks are optional in the same way abort hooks are, and
        # Backend's default registerCompletionHook throws rather than no-opping,
        # so attach() must ask first. A backend without them still records; it
        # only loses the ability to say a collective finished.
        pg = self._init_pg()
        backend = pg._get_backend(self.device)
        self.assertEqual(pg.supports_completion_hooks, self._push_completion)
        self.assertEqual(backend.supports_completion_hooks, self._push_completion)

        before = len(self._hook_entries())
        hook = self._fr_hook(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        entries = self._await_retired(before + 1)[before:]
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["retired"], msg=str(entries[0]))
        # remove() must not throw either: unregisterCompletionHook throws on a
        # backend that has none, so it may only be called if attach() registered
        # one.
        hook.remove()

    def test_profiling_name_carries_backend_name(self):
        # The comm_lib field must be the backend's own name, not a placeholder
        # and not ProcessGroup::getBackendName(), which answers "custom" for
        # anything outside the built-in BackendType enum (nccl2 included).
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        self.assertEqual(len(entries), 1)
        name = entries[0]["profiling_name"]
        self.assertEqual(name, f"{self.backend_name}:all_reduce")
        # The analyzer splits on ":" and expects exactly two fields.
        self.assertEqual(name.count(":"), 1)
        hook.remove()

    def test_inflight_op_is_not_reported_completed(self):
        # Regression test: the hook used to publish an end event that is only
        # recorded when the entry is retired, and query() answers true for an
        # event that was never recorded, so an op still in flight read as
        # "completed" -- and a dump with TORCH_INCLUDE_ONLY_ACTIVE then skipped
        # it, losing exactly the collective a post-mortem is for.
        #
        # An op whose post-hook never runs is what makes that permanent, and a
        # backend that rejects its arguments produces one deterministically:
        # Ops.cpp fires the pre-hook, the backend throws, and firePostHook is
        # never reached (there is no try/catch around it).
        if not self._communicates:
            self.skipTest("needs a backend that validates its arguments")
        pg = self._init_pg()
        hook = self._fr_hook(pg)
        t = torch.ones(4, device=self.device)
        # Warm up so lazy communicator setup does not add entries of its own.
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        before = len(self._all_entries())

        with self.assertRaises(RuntimeError):
            # One output tensor for a world of two.
            dist.all_gather([torch.empty_like(t)], t)

        # The backend validates before it records natively, so the only new
        # entry is the hook's.
        entries = self._all_entries()[before:]
        self.assertEqual(len(entries), 1, msg=str(entries))
        e = entries[0]
        self.assertEqual(e["profiling_name"], self._name("all_gather"))
        self.assertFalse(e["retired"], msg=str(e))
        self.assertNotEqual(e["state"], "completed", msg=str(e))
        self.assertEqual(e["time_discovered_completed_ns"], 0, msg=str(e))
        hook.remove()

    def test_completion_arriving_before_registration_still_retires(self):
        # A Work and its op_id are only ever seen together in the post-hook, so
        # a backend that establishes completion before that runs has nothing to
        # hand the callback to. Dropping it retires nothing and the collective
        # reads "scheduled" for ever -- the false hang this feature exists to
        # rule out -- so the post-hook asks the Work whether it has already
        # finished instead of waiting for a push that is not coming.
        #
        # Constructed, not raced. Post-hooks fire in hook_id order and the
        # flight recorder hook's ids start far above any hand-picked one, so
        # the hook below runs first and holds the window open. It waits on a
        # sentinel collective issued behind the target on the same stream: the
        # backend retires that queue in order, so once the sentinel's entry is
        # retired the target's completion has already been pushed -- into a
        # post-hook that has not run yet, because it is waiting on this one.
        if not self._push_completion:
            self.skipTest("backend pushes no completion")
        pg = self._init_pg()
        pg._enable_collectives_timing()
        hook = self._fr_hook(pg)
        t = torch.ones(1024, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        before = len(self._await_retired())
        observed = {}

        def hold_post_hook_open(args):
            if args.work is None or observed.get("running"):
                return
            observed["running"] = True
            try:
                dist.all_reduce(torch.ones(8, device=self.device))
                deadline = time.time() + 60
                while time.time() < deadline:
                    entries = self._hook_entries()
                    if len(entries) > before + 1 and entries[before + 1]["retired"]:
                        break
                    time.sleep(0.05)
                entries = self._hook_entries()
                observed["target"] = entries[before]
                observed["sentinel"] = entries[before + 1]
            finally:
                observed["running"] = False

        pg.register_post_hook(1, hold_post_hook_open)
        try:
            dist.all_reduce(t)
            torch.cuda.synchronize()
        finally:
            pg.unregister_post_hook(1)

        # The completion really did arrive with no op_id to match it to: had the
        # hook registered the Work first, the same push would have retired the
        # entry before this snapshot was taken.
        sentinel = observed["sentinel"]
        self.assertTrue(sentinel["retired"], msg=str(sentinel))
        target = observed["target"]
        self.assertFalse(target["retired"], msg=str(target))
        self.assertNotEqual(target["state"], "completed", msg=str(target))
        # ... and the post-hook then retires it, rather than leaving a finished
        # collective looking hung for the rest of the job.
        entries = self._await_retired(before + 2)[before:]
        self.assertEqual(len(entries), 2, msg=str(entries))
        e = entries[0]
        self.assertTrue(e["retired"], msg=str(e))
        self.assertEqual(e["state"], "completed", msg=str(e))
        self.assertGreater(e["time_discovered_completed_ns"], 0, msg=str(e))
        # The backend's own measurement is kept, not recomputed or dropped.
        self.assertIn("duration_ms", e, msg=str(e))
        self.assertGreater(e["duration_ms"], 0.0, msg=str(e))
        hook.remove()

    def test_no_recording_during_cuda_graph_capture(self):
        # A collective issued under capture does not run here, it runs at
        # replay, and its Work cannot be polled: querying a CUDA event recorded
        # on a capturing stream invalidates the capture rather than merely
        # failing, and the failure then surfaces from cudaStreamEndCapture. An
        # entry the hook could never observe would read as a collective that
        # never finished, so nothing is recorded at all -- which is also what
        # stock ProcessGroupNCCL does under capture.
        if self.device_type != "cuda":
            self.skipTest("capture test is CUDA only")
        pg = self._init_pg()
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        # Warm up so comm initialization does not happen inside the capture.
        dist.all_reduce(t)
        torch.cuda.synchronize()
        before = len(self._hook_entries())

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(t)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(self._hook_entries()[before:], [])
        # Recording resumes once the capture is over.
        dist.all_reduce(t)
        torch.cuda.synchronize()
        entries = self._await_retired(before + 1)[before:]
        self.assertEqual(len(entries), 1, msg=str(entries))
        self.assertEqual(entries[0]["profiling_name"], self._name("all_reduce"))
        self.assertEqual(entries[0]["state"], "completed", msg=str(entries[0]))
        hook.remove()

    def test_no_recording_of_tensorless_op_during_capture(self):
        # barrier() is the one op that reaches the hook with no tensors at all:
        # Ops.cpp binds a dummy tensor to pick the dispatch key and does not
        # forward it. The capture guard therefore has no device from the op
        # itself and has to take one from the group. Reading "no device" as "not
        # capturing" recorded the barrier, and the post-hook then asked its Work
        # whether it had already completed -- a cudaEventQuery on a capturing
        # stream, which invalidates the capture and surfaces from
        # cudaStreamEndCapture as an unrelated cudaErrorStreamCaptureInvalidated.
        if self.device_type != "cuda":
            self.skipTest("capture test is CUDA only")
        pg = self._init_pg()
        hook = self._fr_hook(pg)

        t = torch.ones(4, device=self.device)
        # Warm up so comm initialization does not happen inside the capture.
        dist.all_reduce(t)
        dist.barrier(device_ids=[self.rank])
        torch.cuda.synchronize()
        before = len(self._hook_entries())

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.barrier(device_ids=[self.rank])
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(self._hook_entries()[before:], [])
        # ... and a tensorless op outside a capture is still recorded, so the
        # guard is not simply refusing every op it cannot get a device from.
        dist.barrier(device_ids=[self.rank])
        torch.cuda.synchronize()
        entries = self._await_retired(before + 1)[before:]
        self.assertEqual(len(entries), 1, msg=str(entries))
        self.assertEqual(entries[0]["profiling_name"], self._name("barrier"))
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


class FlightRecorderHookGlooTest(MultiProcessTestCase):
    """The hook must leave a natively recording backend's ops alone.

    ProcessGroupGloo records in enqueue(), into the very instance the dump APIs
    return by default. Recording gloo ops on top would put two entries with two
    independent collective_seq_ids in the trace for every gloo collective.
    """

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    @staticmethod
    def _entries(**kwargs):
        trace = json.loads(torch._C._distributed_c10d._dump_fr_trace_json(**kwargs))
        return trace.get("entries", [])

    def test_pure_gloo_group_is_not_auto_attached(self):
        # A hook there would record nothing, so do not pay for one.
        pg = self._init_pg()
        self.assertNotIn(pg, _world.pg_flight_recorder_hooks)

    def test_gloo_collectives_recorded_exactly_once(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._entries())

        t = torch.ones(8)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)

        entries = self._entries()[before:]
        self.assertEqual(
            [e["profiling_name"] for e in entries],
            ["gloo:all_reduce", "gloo:broadcast"],
        )
        # Only the hook fills duration_ms in; gloo's native recording retires
        # with compute_duration=false and leaves the field out.
        for e in entries:
            self.assertNotIn("duration_ms", e, msg=str(e))
        hook.remove()

    def test_default_dump_returns_the_gloo_instance(self):
        pg = self._init_pg()
        FlightRecorderHook.attach(pg)
        dist.all_reduce(torch.ones(8))

        default = pickle.loads(torch._C._distributed_c10d._dump_fr_trace())
        named = pickle.loads(torch._C._distributed_c10d._dump_fr_trace(backend="gloo"))
        self.assertEqual(
            [e["profiling_name"] for e in default["entries"]],
            [e["profiling_name"] for e in named["entries"]],
        )
        self.assertIn(
            "gloo:all_reduce", [e["profiling_name"] for e in named["entries"]]
        )


@unittest.skipIf(
    not TEST_CUDA or torch.cuda.device_count() < 2,
    "mixed backend tests require at least 2 GPUs",
)
@unittest.skipIf(
    not dist.is_backend_available("nccl2"), "nccl2 backend is not available"
)
class FlightRecorderHookMixedBackendTest(MultiProcessTestCase):
    """A group whose CPU half records itself and whose CUDA half does not."""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @staticmethod
    def _entries(**kwargs):
        trace = json.loads(torch._C._distributed_c10d._dump_fr_trace_json(**kwargs))
        return trace.get("entries", [])

    def test_mixed_group_records_the_cuda_half_only(self):
        torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl2",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        pg = dist.group.WORLD
        # The whole-group skip used to lose the nccl2 half of a mixed group;
        # the hook now filters per op, so it is attached again.
        self.assertIn(pg, _world.pg_flight_recorder_hooks)

        before_gloo = len(self._entries())
        before_nccl2 = len(self._entries(backend="nccl2"))
        dist.all_reduce(torch.ones(8))
        dist.all_reduce(torch.ones(16, device=f"cuda:{self.rank}"))
        torch.cuda.synchronize()

        # The CPU collective is gloo's own entry, recorded once, and the CUDA
        # one is the hook's, in nccl2's instance.
        gloo_entries = self._entries()[before_gloo:]
        self.assertEqual(
            [(e["profiling_name"], e["input_sizes"]) for e in gloo_entries],
            [("gloo:all_reduce", [[8]])],
        )
        nccl2_entries = self._entries(backend="nccl2")[before_nccl2:]
        self.assertEqual(
            [(e["profiling_name"], e["input_sizes"]) for e in nccl2_entries],
            [("nccl2:all_reduce", [[16]])],
        )

    def test_two_hooked_backends_do_not_share_a_buffer(self):
        torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl2",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        fake_pg = dist.new_group(backend="fake")
        fake_hook = FlightRecorderHook.attach(
            fake_pg, dist.get_process_group_ranks(fake_pg)
        )

        before_nccl2 = len(self._entries(backend="nccl2"))
        before_fake = len(self._entries(backend="fake"))
        dist.all_reduce(torch.ones(16, device=f"cuda:{self.rank}"))
        torch.cuda.synchronize()
        dist.all_reduce(torch.ones(8), group=fake_pg)

        self.assertEqual(
            [
                e["profiling_name"]
                for e in self._entries(backend="nccl2")[before_nccl2:]
            ],
            ["nccl2:all_reduce"],
        )
        self.assertEqual(
            [e["profiling_name"] for e in self._entries(backend="fake")[before_fake:]],
            ["fake:all_reduce"],
        )
        # pg_ids come from one process-wide counter, so two hooked groups never
        # collide even if they did share an instance.
        pg_ids = {
            self._entries(backend="nccl2")[-1]["pg_id"],
            self._entries(backend="fake")[-1]["pg_id"],
        }
        self.assertEqual(len(pg_ids), 2)
        fake_hook.remove()


@unittest.skipIf(
    not TEST_CUDA or torch.cuda.device_count() < 2,
    "default nccl backend tests require at least 2 GPUs",
)
@unittest.skipIf(not dist.is_backend_available("nccl"), "nccl backend is not available")
class FlightRecorderHookDefaultNcclTest(MultiProcessTestCase):
    """The "nccl" name does not always build the same thing.

    TORCH_DIST_USE_NCCL2 decides whether it is stock ProcessGroupNCCL, which
    feeds a FlightRecorder by itself, or nccl2, which is invisible without the
    hook. Whichever one the name resolved to is what the auto-attach has to
    follow: skipping a group that turned out to be nccl2 leaves the default
    backend with no flight recorder at all, and says nothing about it.
    """

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("TORCH_DIST_USE_NCCL2", None)
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @staticmethod
    def _entries(backend):
        trace = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(backend=backend)
        )
        return trace.get("entries", [])

    def _init_nccl(self, use_nccl2):
        # Read once, where the backend is registered, which is the first time a
        # group asks for "nccl" -- so it has to be set before init.
        os.environ["TORCH_DIST_USE_NCCL2"] = use_nccl2
        torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        pg = dist.group.WORLD
        # The backend's own name, which is what decides whether it records
        # itself. Asserting on it rather than on the env var is the point: the
        # default this variable implies has changed before.
        backend = pg._get_backend(torch.device("cuda", self.rank))
        return pg, backend.name()

    def test_nccl_name_built_on_nccl2_is_hooked(self):
        pg, name = self._init_nccl("1")
        self.assertEqual(name, "nccl2")
        self.assertIn(pg, _world.pg_flight_recorder_hooks)

        before = len(self._entries("nccl2"))
        dist.all_reduce(torch.ones(8, device=f"cuda:{self.rank}"))
        torch.cuda.synchronize()
        self.assertEqual(
            [e["profiling_name"] for e in self._entries("nccl2")[before:]],
            ["nccl2:all_reduce"],
        )

    def test_nccl_name_built_on_stock_is_not_hooked(self):
        pg, name = self._init_nccl("0")
        self.assertEqual(name, "nccl")
        self.assertNotIn(pg, _world.pg_flight_recorder_hooks)

        before = len(self._entries("nccl"))
        dist.all_reduce(torch.ones(8, device=f"cuda:{self.rank}"))
        torch.cuda.synchronize()
        # Nothing in the hook's instance, and the native recording that made
        # the hook unnecessary is still there.
        self.assertEqual(self._entries("nccl")[before:], [])
        native = json.loads(torch._C._distributed_c10d._dump_nccl_trace_json())
        names = [e["profiling_name"] for e in native.get("entries", [])]
        self.assertIn("nccl:all_reduce", names)


if __name__ == "__main__":
    run_tests()
