# Owner(s): ["oncall: distributed"]
"""Integration tests for torchmux and vnccl.

Tests cooperative scheduling trace ordering, collective correctness
under torchmux (file-based PG), and vnccl (thread-based PG).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


TRAIN_SCRIPT = """
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)

# Two allreduces with compute in between
x = torch.ones(4, 4, device=device)
dist.all_reduce(x)
assert torch.allclose(x, torch.full_like(x, float(ws))), f"first allreduce failed: {x}"
y = x @ x
dist.all_reduce(y)
expected_y = torch.full_like(y, float(ws * ws * ws * 4))
assert torch.allclose(y, expected_y), f"second allreduce failed: {y}"

dist.destroy_process_group()
"""


CORRECTNESS_SCRIPT = """
import json
import os
import sys
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)

results = {}

# allreduce: each rank contributes (rank+1), expect sum = ws*(ws+1)/2
x = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(x)
expected_sum = ws * (ws + 1) / 2
results["allreduce"] = torch.allclose(x, torch.full_like(x, expected_sum))

# broadcast: root=0 sends tensor, others receive
y = torch.full((4,), 42.0, device=device) if rank == 0 else torch.zeros(4, device=device)
dist.broadcast(y, src=0)
results["broadcast"] = torch.allclose(y, torch.full_like(y, 42.0))

# allgather: each rank contributes rank, gather all
inp = torch.full((2,), float(rank), device=device)
gathered = [torch.zeros(2, device=device) for _ in range(ws)]
dist.all_gather(gathered, inp)
allgather_ok = all(
    torch.allclose(gathered[r], torch.full((2,), float(r), device=device))
    for r in range(ws)
)
results["allgather"] = allgather_ok

# reduce_scatter: each rank contributes full tensor, gets a chunk of the sum
full = torch.arange(ws * 2, dtype=torch.float32, device=device)
out = torch.zeros(2, device=device)
dist.reduce_scatter_tensor(out, full)
chunk = full[rank * 2 : (rank + 1) * 2] * ws
results["reduce_scatter"] = torch.allclose(out, chunk)

# barrier: just verify it completes without hanging
dist.barrier()
results["barrier"] = True

dist.destroy_process_group()

output_path = os.path.join(os.environ["TORCHMUX_RESULTS_DIR"], f"rank{rank}.json")
with open(output_path, "w") as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxTraceOrdering(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)
        self._script = os.path.join(self._tmpdir, "train.py")
        with open(self._script, "w") as f:
            f.write(TRAIN_SCRIPT)

    def _run_torchmux(self, nproc, script, env_extra=None):
        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = trace_dir
        env["TORCHMUX_NGPUS"] = "1"
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", str(nproc), script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        return trace_dir

    def test_3worker_trace_ordering(self):
        """Assert trace events follow cooperative scheduling invariants.

        Expected pattern per collective (3 workers):
          W0: compute -> collective_start -> snapshot
          W1: compute -> collective_start -> snapshot
          W2: compute -> collective_start -> collective_finish (resolves)
          W2: continues to next compute...
          W2: next_collective_start -> snapshot (yields for first time)
          W0: restore -> collective_finish -> compute
          ...
        """
        trace_dir = self._run_torchmux(3, self._script)

        natural_path = os.path.join(trace_dir, "torchmux_natural.json")
        self.assertTrue(os.path.exists(natural_path))

        with open(natural_path) as f:
            trace = json.load(f)

        events = [
            e for e in trace["traceEvents"] if e.get("ph") == "X"
        ]
        self.assertGreater(len(events), 0, "no trace events")

        per_worker = {}
        for e in events:
            per_worker.setdefault(e["pid"], []).append(e)
        for pid in per_worker:
            per_worker[pid].sort(key=lambda e: e["ts"])

        self.assertEqual(len(per_worker), 3, "expected 3 workers")

        # Global timeline: sort all events by start time
        timeline = sorted(events, key=lambda e: e["ts"])

        # -- Invariant 1: compute spans never overlap across workers --
        compute_spans = [
            (e["pid"], e["ts"], e["ts"] + e["dur"])
            for e in timeline
            if e["cat"] == "compute"
        ]
        for i in range(len(compute_spans)):
            for j in range(i + 1, len(compute_spans)):
                pi, si, ei = compute_spans[i]
                pj, sj, ej = compute_spans[j]
                if pi != pj:
                    self.assertFalse(
                        si < ej and sj < ei,
                        f"compute overlap: worker {pi} [{si}-{ei}] "
                        f"vs worker {pj} [{sj}-{ej}]",
                    )

        # -- Invariant 2: every snapshot is preceded by either a
        #    collective or a compute (never another snapshot) --
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "mux" and e["name"] == "snapshot":
                    if i > 0:
                        prev = worker_events[i - 1]["cat"]
                        self.assertIn(
                            prev,
                            ("collective", "compute"),
                            f"worker {pid}: snapshot preceded by "
                            f"{prev} (expected collective or compute)",
                        )

        # -- Invariant 3: every restore is followed by either a
        #    collective or a compute (never another restore) --
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "mux" and e["name"] == "restore":
                    if i < len(worker_events) - 1:
                        nxt = worker_events[i + 1]["cat"]
                        self.assertIn(
                            nxt,
                            ("collective", "compute"),
                            f"worker {pid}: restore followed by "
                            f"{nxt} (expected collective or compute)",
                        )

        # -- Invariant 4: the last worker to arrive at a collective
        #    resolves without snapshotting. At least one worker must
        #    have at least one collective NOT followed by a snapshot. --
        any_immediate = False
        for pid, worker_events in per_worker.items():
            for i, e in enumerate(worker_events):
                if e["cat"] == "collective":
                    next_is_snapshot = (
                        i + 1 < len(worker_events)
                        and worker_events[i + 1]["cat"] == "mux"
                        and worker_events[i + 1]["name"] == "snapshot"
                    )
                    if not next_is_snapshot:
                        any_immediate = True
        self.assertTrue(
            any_immediate,
            "every collective across all workers is followed by a "
            "snapshot (expected the last-to-arrive worker to resolve "
            "at least one collective immediately)",
        )

        # -- Invariant 5: snapshot-restore pairs bracket gaps where
        #    other workers run. No worker should have snapshot without
        #    a later restore (except the final cleanup). --
        for pid, worker_events in per_worker.items():
            snapshots = sum(
                1
                for e in worker_events
                if e["cat"] == "mux" and e["name"] == "snapshot"
            )
            restores = sum(
                1
                for e in worker_events
                if e["cat"] == "mux" and e["name"] == "restore"
            )
            self.assertIn(
                snapshots - restores,
                (0, 1),
                f"worker {pid}: {snapshots} snapshots vs {restores} "
                f"restores (expected equal or off by one)",
            )

        # -- Validate synthetic trace is also produced and well-formed --
        synthetic_path = os.path.join(trace_dir, "torchmux_synthetic.json")
        self.assertTrue(os.path.exists(synthetic_path))

        with open(synthetic_path) as f:
            synthetic = json.load(f)

        syn_events = [
            e for e in synthetic["traceEvents"] if e.get("ph") == "X"
        ]
        self.assertGreater(len(syn_events), 0, "no synthetic trace events")

        for e in syn_events:
            for field in ("cat", "name", "pid", "tid", "ts", "dur"):
                self.assertIn(
                    field, e, f"synthetic event missing '{field}': {e}"
                )
            self.assertGreaterEqual(e["ts"], 0)
            self.assertGreaterEqual(e["dur"], 0)

        syn_mux = [e for e in syn_events if e["cat"] == "mux"]
        self.assertEqual(
            len(syn_mux), 0,
            "synthetic trace should not contain snapshot/restore events",
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxCollectiveCorrectness(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_collective_correctness_2workers(self):
        script_path = os.path.join(self._tmpdir, "correctness.py")
        with open(script_path, "w") as f:
            f.write(CORRECTNESS_SCRIPT)

        results_dir = os.path.join(self._tmpdir, "results")
        os.makedirs(results_dir)

        env = os.environ.copy()
        env["TORCHMUX_NGPUS"] = "1"
        env["TORCHMUX_RESULTS_DIR"] = results_dir
        env["TORCHMUX_TRACE_DIR"] = self._tmpdir

        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", script_path,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

        for rank in range(2):
            rpath = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(rpath), f"missing results for rank {rank}")
            with open(rpath) as f:
                results = json.load(f)
            for op_name, passed in results.items():
                self.assertTrue(passed, f"rank {rank}: {op_name} failed")


class TestTorchmuxSyntheticTrace(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_synthetic_trace_structure(self):
        from torch.distributed.torchmux_trace import export_synthetic

        events_by_rank = {
            0: [
                ("compute", "compute", 0, 100),
                ("collective", "allreduce", 100, 10),
                ("compute", "compute", 110, 50),
                ("collective", "allreduce", 160, 10),
            ],
            1: [
                ("compute", "compute", 200, 80),
                ("collective", "allreduce", 280, 10),
                ("compute", "compute", 290, 60),
                ("collective", "allreduce", 350, 10),
            ],
        }

        path = os.path.join(self._tmpdir, "synthetic.json")
        export_synthetic(events_by_rank, path, 2)

        with open(path) as f:
            trace = json.load(f)

        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertGreater(len(x_events), 0)

        compute_events = [e for e in x_events if e["cat"] == "compute"]
        coll_events = [e for e in x_events if e["cat"] == "collective"]

        self.assertEqual(len(compute_events), 4)
        self.assertEqual(len(coll_events), 4)

        for rank in (0, 1):
            rank_computes = [
                e for e in compute_events if e["pid"] == rank
            ]
            rank_colls = [e for e in coll_events if e["pid"] == rank]
            self.assertEqual(len(rank_computes), 2)
            self.assertEqual(len(rank_colls), 2)

        # In the synthetic trace, both workers' first compute starts
        # at ts=0 (overlapped)
        first_computes = [
            e for e in compute_events
            if e["ts"] == 0
        ]
        self.assertEqual(len(first_computes), 2)

        # No snapshot/restore events in synthetic trace
        mux_events = [e for e in x_events if e["cat"] == "mux"]
        self.assertEqual(len(mux_events), 0)

    def test_natural_trace_preserves_timestamps(self):
        from torch.distributed.torchmux_trace import export_natural

        events_by_rank = {
            0: [
                ("compute", "compute", 1000, 100),
                ("mux", "snapshot", 1100, 5),
            ],
            1: [
                ("compute", "compute", 1200, 80),
            ],
        }

        path = os.path.join(self._tmpdir, "natural.json")
        export_natural(events_by_rank, path, 2)

        with open(path) as f:
            trace = json.load(f)

        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertEqual(len(x_events), 3)

        # First event should be at ts=0 (base_ts subtracted)
        ts_values = sorted(e["ts"] for e in x_events)
        self.assertEqual(ts_values[0], 0)

    def test_empty_events(self):
        from torch.distributed.torchmux_trace import export_natural, export_synthetic

        path_n = os.path.join(self._tmpdir, "empty_natural.json")
        path_s = os.path.join(self._tmpdir, "empty_synthetic.json")
        export_natural({}, path_n, 2)
        export_synthetic({}, path_s, 2)

        for path in (path_n, path_s):
            with open(path) as f:
                trace = json.load(f)
            x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
            self.assertEqual(len(x_events), 0)


class TestMuxDevice(TestCase):
    def test_cuda_index_remapping(self):
        from torch.distributed.torchmux import _MuxDevice, _OrigDevice

        import torch.distributed.torchmux as tm
        orig_ngpus = tm._ngpus
        try:
            tm._ngpus = 2
            d = _MuxDevice("cuda:5")
            self.assertEqual(d, _OrigDevice("cuda:1"))

            d = _MuxDevice("cuda:4")
            self.assertEqual(d, _OrigDevice("cuda:0"))

            d = _MuxDevice("cuda:0")
            self.assertEqual(d, _OrigDevice("cuda:0"))
        finally:
            tm._ngpus = orig_ngpus

    def test_non_cuda_passthrough(self):
        from torch.distributed.torchmux import _MuxDevice, _OrigDevice

        d = _MuxDevice("cpu")
        self.assertEqual(d, _OrigDevice("cpu"))

    def test_isinstance_compat(self):
        from torch.distributed.torchmux import _MuxDevice, _OrigDevice

        d = _MuxDevice("cpu")
        self.assertIsInstance(d, _OrigDevice)

    def test_cuda_no_index_passthrough(self):
        from torch.distributed.torchmux import _MuxDevice, _OrigDevice

        import torch.distributed.torchmux as tm
        orig_ngpus = tm._ngpus
        try:
            tm._ngpus = 2
            d = _MuxDevice("cuda")
            self.assertEqual(d, _OrigDevice("cuda"))
            self.assertIsNone(d.index)
        finally:
            tm._ngpus = orig_ngpus

    def test_remapping_single_gpu(self):
        from torch.distributed.torchmux import _MuxDevice, _OrigDevice

        import torch.distributed.torchmux as tm
        orig_ngpus = tm._ngpus
        try:
            tm._ngpus = 1
            for idx in range(8):
                d = _MuxDevice(f"cuda:{idx}")
                self.assertEqual(d, _OrigDevice("cuda:0"))
        finally:
            tm._ngpus = orig_ngpus


class _VNCCLTestBase(TestCase):
    """Shared base class for vnccl tests with common setup/teardown."""

    def setUp(self):
        super().setUp()
        self._world_size = 3
        self._pgs = []

    def tearDown(self):
        import torch.distributed.vnccl as vnccl
        from torch.distributed.vnccl import VNCCLProcessGroup

        VNCCLProcessGroup._active.clear()
        for pg in self._pgs:
            dist.distributed_c10d._world.pg_names.pop(pg, None)
        self._pgs = []
        vnccl._next_rank = 0
        vnccl._rng_states.clear()
        super().tearDown()

    def _make_pgs(self, pg_name="vnccl_test"):
        from torch.distributed.vnccl import VNCCLProcessGroup

        pgs = []
        for r in range(self._world_size):
            pg = VNCCLProcessGroup(r, self._world_size)
            dist.distributed_c10d._world.pg_names[pg] = pg_name
            pgs.append(pg)
        self._pgs = pgs
        return pgs

    def _run_on_pgs(self, pgs, fn):
        results = [None] * self._world_size
        errors = [None] * self._world_size

        def _worker(rank):
            try:
                results[rank] = fn(rank, pgs[rank])
            except Exception as e:
                errors[rank] = e

        threads = [
            threading.Thread(target=_worker, args=(r,))
            for r in range(self._world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        hung = [r for r, t in enumerate(threads) if t.is_alive()]
        if hung:
            raise RuntimeError(
                f"ranks {hung} still alive after 30s timeout"
            )
        for r, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f"rank {r} failed") from e
        return results


class TestVNCCL(_VNCCLTestBase):
    """Tests vnccl collective correctness using direct PG method calls.

    Each test creates VNCCLProcessGroup instances directly and runs
    collective operations from separate threads, bypassing
    dist.init_process_group to avoid global state conflicts.
    """

    def test_allreduce_sum(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_allreduce_avg(self):
        from torch._C._distributed_c10d import AllreduceOptions, ReduceOp

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            opts = AllreduceOptions()
            opts.reduceOp = ReduceOp.AVG
            pg.allreduce(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = (self._world_size + 1) / 2.0
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_allreduce_product(self):
        from torch._C._distributed_c10d import AllreduceOptions, ReduceOp

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            opts = AllreduceOptions()
            opts.reduceOp = ReduceOp.PRODUCT
            pg.allreduce(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = 1.0
        for r in range(self._world_size):
            expected *= r + 1
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_allreduce_min(self):
        from torch._C._distributed_c10d import AllreduceOptions, ReduceOp

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            opts = AllreduceOptions()
            opts.reduceOp = ReduceOp.MIN
            pg.allreduce(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), 1.0))

    def test_allreduce_max(self):
        from torch._C._distributed_c10d import AllreduceOptions, ReduceOp

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            opts = AllreduceOptions()
            opts.reduceOp = ReduceOp.MAX
            pg.allreduce(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), float(self._world_size)))

    def test_allreduce_bitwise_ops(self):
        from torch._C._distributed_c10d import AllreduceOptions, ReduceOp

        pgs = self._make_pgs()

        for op, expected_fn in [
            (ReduceOp.BAND, lambda vals: vals[0] & vals[1] & vals[2]),
            (ReduceOp.BOR, lambda vals: vals[0] | vals[1] | vals[2]),
            (ReduceOp.BXOR, lambda vals: vals[0] ^ vals[1] ^ vals[2]),
        ]:
            pgs = self._make_pgs()

            def _work(rank, pg, _op=op):
                t = [torch.tensor([rank + 1, rank + 10], dtype=torch.int64)]
                opts = AllreduceOptions()
                opts.reduceOp = _op
                pg.allreduce(t, opts)
                return t[0]

            results = self._run_on_pgs(pgs, _work)
            vals = [torch.tensor([r + 1, r + 10], dtype=torch.int64) for r in range(self._world_size)]
            expected = expected_fn(vals)
            for r, t in enumerate(results):
                self.assertEqual(t, expected)

    def test_allreduce_coalesced(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce_coalesced(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_broadcast(self):
        pgs = self._make_pgs()

        from torch._C._distributed_c10d import BroadcastOptions

        def _work(rank, pg):
            t = [torch.full((4,), 99.0) if rank == 0 else torch.zeros(4)]
            opts = BroadcastOptions()
            opts.rootRank = 0
            pg.broadcast(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), 99.0))

    def test_broadcast_non_root(self):
        pgs = self._make_pgs()

        from torch._C._distributed_c10d import BroadcastOptions

        def _work(rank, pg):
            t = [torch.full((4,), 77.0) if rank == 2 else torch.zeros(4)]
            opts = BroadcastOptions()
            opts.rootRank = 2
            pg.broadcast(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), 77.0))

    def test_allgather(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = [torch.full((2,), float(rank))]
            out = [[torch.zeros(2) for _ in range(self._world_size)]]
            pg.allgather(out, inp)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        for r, gathered in enumerate(results):
            for src, t in enumerate(gathered):
                self.assertEqual(t, torch.full((2,), float(src)))

    def test_allgather_base(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = torch.full((2,), float(rank))
            out = torch.zeros(2 * self._world_size)
            pg._allgather_base(out, inp)
            return out

        results = self._run_on_pgs(pgs, _work)
        expected = torch.cat([torch.full((2,), float(r)) for r in range(self._world_size)])
        for r, t in enumerate(results):
            self.assertEqual(t, expected)

    def test_reduce_scatter(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            chunks = [torch.full((2,), float(rank + 1)) for _ in range(self._world_size)]
            out = [torch.zeros(2)]
            pg.reduce_scatter(out, [chunks])
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), expected))

    def test_reduce_scatter_base(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = torch.full((2 * self._world_size,), float(rank + 1))
            out = torch.zeros(2)
            pg._reduce_scatter_base(out, inp)
            return out

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), expected))

    def test_scatter(self):
        pgs = self._make_pgs()

        from torch._C._distributed_c10d import ScatterOptions

        def _work(rank, pg):
            if rank == 0:
                src = [torch.full((2,), float(i)) for i in range(self._world_size)]
            else:
                src = []
            out = [torch.zeros(2)]
            opts = ScatterOptions()
            opts.rootRank = 0
            pg.scatter(out, [src], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), float(r)))

    def test_gather(self):
        pgs = self._make_pgs()

        from torch._C._distributed_c10d import ScatterOptions

        def _work(rank, pg):
            inp = [torch.full((2,), float(rank))]
            if rank == 0:
                out = [[torch.zeros(2) for _ in range(self._world_size)]]
            else:
                out = [[]]
            opts = ScatterOptions()
            opts.rootRank = 0
            pg.gather(out, inp, opts)
            return out[0] if rank == 0 else None

        results = self._run_on_pgs(pgs, _work)
        gathered = results[0]
        for src, t in enumerate(gathered):
            self.assertEqual(t, torch.full((2,), float(src)))

    def test_alltoall(self):
        pgs = self._make_pgs()
        ws = self._world_size

        def _work(rank, pg):
            inp = [torch.full((2,), float(rank * ws + dst)) for dst in range(ws)]
            out = [torch.zeros(2) for _ in range(ws)]
            pg.alltoall(out, inp)
            return out

        results = self._run_on_pgs(pgs, _work)
        for dst in range(ws):
            for src in range(ws):
                self.assertEqual(
                    results[dst][src],
                    torch.full((2,), float(src * ws + dst)),
                )

    def test_barrier(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            pg.barrier()
            return True

        results = self._run_on_pgs(pgs, _work)
        for r, v in enumerate(results):
            self.assertTrue(v, f"rank {r}: barrier failed")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_allreduce_cuda(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1), device="cuda")]
            pg.allreduce(t)
            return t[0].cpu()

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_multiple_collectives_in_sequence(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            results = []
            for i in range(5):
                t = [torch.full((4,), float(rank + i))]
                pg.allreduce(t)
                results.append(t[0].clone())
            return results

        results = self._run_on_pgs(pgs, _work)
        for i in range(5):
            expected = sum(r + i for r in range(self._world_size))
            for r in range(self._world_size):
                self.assertEqual(
                    results[r][i],
                    torch.full((4,), float(expected)),
                )

    def test_single_element_tensor(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.tensor([float(rank + 1)])]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.tensor([expected]))

    def test_float64_dtype(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1), dtype=torch.float64)]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t.dtype, torch.float64)
            self.assertEqual(t, torch.full((4,), expected, dtype=torch.float64))

    def test_int_dtype_with_sum(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), rank + 1, dtype=torch.int64)]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) // 2
        for r, t in enumerate(results):
            self.assertEqual(t.dtype, torch.int64)
            self.assertEqual(t, torch.full((4,), expected, dtype=torch.int64))

    def test_world_size_2(self):
        self._world_size = 2
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = 3.0
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))


class TestVNCCLReduceScatterOps(_VNCCLTestBase):
    """Verify vnccl reduce_scatter works with non-SUM reduce ops."""

    def test_reduce_scatter_product(self):
        from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            chunks = [torch.full((2,), float(rank + 1)) for _ in range(self._world_size)]
            out = [torch.zeros(2)]
            opts = ReduceScatterOptions()
            opts.reduceOp = ReduceOp.PRODUCT
            pg.reduce_scatter(out, [chunks], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        expected = 1.0
        for r in range(self._world_size):
            expected *= r + 1
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), expected))

    def test_reduce_scatter_max(self):
        from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            chunks = [torch.full((2,), float(rank + 1)) for _ in range(self._world_size)]
            out = [torch.zeros(2)]
            opts = ReduceScatterOptions()
            opts.reduceOp = ReduceOp.MAX
            pg.reduce_scatter(out, [chunks], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        expected = float(self._world_size)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), expected))

    def test_reduce_scatter_min(self):
        from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            chunks = [torch.full((2,), float(rank + 1)) for _ in range(self._world_size)]
            out = [torch.zeros(2)]
            opts = ReduceScatterOptions()
            opts.reduceOp = ReduceOp.MIN
            pg.reduce_scatter(out, [chunks], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), 1.0))

    def test_reduce_scatter_avg(self):
        from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            chunks = [torch.full((2,), float(rank + 1)) for _ in range(self._world_size)]
            out = [torch.zeros(2)]
            opts = ReduceScatterOptions()
            opts.reduceOp = ReduceOp.AVG
            pg.reduce_scatter(out, [chunks], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        expected = (self._world_size + 1) / 2.0
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((2,), expected))


class TestVNCCLCoalesced(_VNCCLTestBase):
    """Test coalesced operations that FSDP2 relies on."""

    def test_allgather_into_tensor_coalesced(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inputs = [
                torch.full((2,), float(rank)),
                torch.full((3,), float(rank + 10)),
            ]
            outputs = [
                torch.zeros(2 * self._world_size),
                torch.zeros(3 * self._world_size),
            ]
            pg.allgather_into_tensor_coalesced(outputs, inputs)
            return outputs

        results = self._run_on_pgs(pgs, _work)
        for r, outputs in enumerate(results):
            expected_0 = torch.cat(
                [torch.full((2,), float(src)) for src in range(self._world_size)]
            )
            expected_1 = torch.cat(
                [torch.full((3,), float(src + 10)) for src in range(self._world_size)]
            )
            self.assertEqual(outputs[0], expected_0)
            self.assertEqual(outputs[1], expected_1)

    def test_reduce_scatter_tensor_coalesced(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            inputs = [
                torch.full((2 * self._world_size,), float(rank + 1)),
                torch.full((3 * self._world_size,), float(rank + 1)),
            ]
            outputs = [torch.zeros(2), torch.zeros(3)]
            pg.reduce_scatter_tensor_coalesced(outputs, inputs)
            return outputs

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, outputs in enumerate(results):
            self.assertEqual(outputs[0], torch.full((2,), expected))
            self.assertEqual(outputs[1], torch.full((3,), expected))


class TestVNCCLWorldSize1(_VNCCLTestBase):
    """Verify vnccl works with a single rank."""

    def test_allreduce_single_rank(self):
        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), 42.0)]
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0], torch.full((4,), 42.0))

    def test_barrier_single_rank(self):
        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            pg.barrier()
            return True

        results = self._run_on_pgs(pgs, _work)
        self.assertTrue(results[0])

    def test_allgather_single_rank(self):
        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = [torch.full((2,), 7.0)]
            out = [[torch.zeros(2)]]
            pg.allgather(out, inp)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0][0], torch.full((2,), 7.0))

    def test_broadcast_single_rank(self):
        from torch._C._distributed_c10d import BroadcastOptions

        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), 55.0)]
            opts = BroadcastOptions()
            opts.rootRank = 0
            pg.broadcast(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0], torch.full((4,), 55.0))

    def test_reduce_scatter_single_rank(self):
        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = torch.full((2,), 5.0)
            out = torch.zeros(2)
            pg._reduce_scatter_base(out, inp)
            return out

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0], torch.full((2,), 5.0))

    def test_scatter_single_rank(self):
        from torch._C._distributed_c10d import ScatterOptions

        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            src = [torch.full((2,), 3.0)]
            out = [torch.zeros(2)]
            opts = ScatterOptions()
            opts.rootRank = 0
            pg.scatter(out, [src], opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0], torch.full((2,), 3.0))

    def test_gather_single_rank(self):
        from torch._C._distributed_c10d import ScatterOptions

        self._world_size = 1
        pgs = self._make_pgs()

        def _work(rank, pg):
            inp = [torch.full((2,), 9.0)]
            out = [[torch.zeros(2)]]
            opts = ScatterOptions()
            opts.rootRank = 0
            pg.gather(out, inp, opts)
            return out[0]

        results = self._run_on_pgs(pgs, _work)
        self.assertEqual(results[0][0], torch.full((2,), 9.0))


class TestVNCCLNonContiguous(_VNCCLTestBase):
    """Verify vnccl handles non-contiguous tensor inputs."""

    def test_allreduce_non_contiguous(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            base = torch.full((4, 4), float(rank + 1))
            t = [base[:, 0]]
            self.assertFalse(t[0].is_contiguous())
            pg.allreduce(t)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        expected = self._world_size * (self._world_size + 1) / 2
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), expected))

    def test_broadcast_non_contiguous(self):
        from torch._C._distributed_c10d import BroadcastOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            if rank == 0:
                base = torch.full((4, 4), 99.0)
                t = [base[:, 0]]
            else:
                t = [torch.zeros(4)]
            opts = BroadcastOptions()
            opts.rootRank = 0
            pg.broadcast(t, opts)
            return t[0]

        results = self._run_on_pgs(pgs, _work)
        for r, t in enumerate(results):
            self.assertEqual(t, torch.full((4,), 99.0))


class TestMuxPGNotImplemented(TestCase):
    def test_scatter_raises(self):
        from torch.distributed.torchmux import _MuxPG

        pg = _MuxPG(0, 2)
        with self.assertRaisesRegex(NotImplementedError, "_MuxPG.scatter"):
            pg.scatter(None, None)

    def test_gather_raises(self):
        from torch.distributed.torchmux import _MuxPG

        pg = _MuxPG(0, 2)
        with self.assertRaisesRegex(NotImplementedError, "_MuxPG.gather"):
            pg.gather(None, None)

    def test_alltoall_raises(self):
        from torch.distributed.torchmux import _MuxPG

        pg = _MuxPG(0, 2)
        with self.assertRaisesRegex(NotImplementedError, "_MuxPG.alltoall"):
            pg.alltoall(None, None)


class TestSharedInt(TestCase):
    def test_basic_read_write(self):
        from torch.distributed.torchmux import _SharedInt

        si = _SharedInt()
        self.addCleanup(si.cleanup)
        self.assertEqual(si.value, 0)
        si.value = 42
        self.assertEqual(si.value, 42)
        si.value = -1
        self.assertEqual(si.value, -1)

    def test_pickle_roundtrip(self):
        import pickle

        from torch.distributed.torchmux import _SharedInt

        si = _SharedInt()
        self.addCleanup(si.cleanup)
        si.value = 123

        data = pickle.dumps(si)
        si2 = pickle.loads(data)
        self.assertEqual(si2.value, 123)
        si2.value = 456
        self.assertEqual(si.value, 456)

    def test_cleanup_then_del_no_error(self):
        from torch.distributed.torchmux import _SharedInt

        si = _SharedInt()
        si.cleanup()
        del si


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxSingleWorker(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_nproc_1(self):
        script = os.path.join(self._tmpdir, "single.py")
        with open(script, "w") as f:
            f.write(
                "import os, torch, torch.distributed as dist\n"
                "dist.init_process_group()\n"
                "assert dist.get_rank() == 0\n"
                "assert dist.get_world_size() == 1\n"
                "device = f\"cuda:{0 % int(os.environ.get('TORCHMUX_NGPUS', '1'))}\"\n"
                "torch.cuda.set_device(device)\n"
                "x = torch.ones(4, device=device)\n"
                "dist.all_reduce(x)\n"
                "assert torch.allclose(x, torch.ones(4, device=device))\n"
                "dist.destroy_process_group()\n"
            )
        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir)
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = trace_dir
        env["TORCHMUX_NGPUS"] = "1"
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "1", script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux nproc=1 failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxErrorHandling(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_worker_crash_propagates(self):
        script = os.path.join(self._tmpdir, "crash.py")
        with open(script, "w") as f:
            f.write(
                "import os, torch, torch.distributed as dist\n"
                "dist.init_process_group()\n"
                "rank = dist.get_rank()\n"
                "device = f\"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}\"\n"
                "torch.cuda.set_device(device)\n"
                "raise RuntimeError('intentional crash')\n"
            )
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = self._tmpdir
        env["TORCHMUX_NGPUS"] = "1"
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)


class TestVNCCLErrorPropagation(_VNCCLTestBase):
    """Verify vnccl handles resolver errors without deadlocking."""

    def test_resolver_error_propagates_to_all_ranks(self):
        from torch.distributed.vnccl import VNCCLProcessGroup

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            if rank == self._world_size - 1:
                t = [torch.full((4,), float("nan"))]
                t[0] = t[0].to(torch.int32)
                t[0] = t[0].view(torch.float32)
            pg.allreduce(t)
            return t[0]

        class _BrokenAllReduce:
            def __init__(self, op):
                self.op = op.op

            def work(self, data):
                raise ValueError("intentional resolver failure")

        orig_do = VNCCLProcessGroup._do

        def _patched_do(self_pg, op, data):
            if hasattr(op, "op"):
                op = _BrokenAllReduce(op)
            return orig_do(self_pg, op, data)

        VNCCLProcessGroup._do = _patched_do
        try:
            with self.assertRaises(RuntimeError):
                self._run_on_pgs(pgs, _work)
        finally:
            VNCCLProcessGroup._do = orig_do

    def test_active_cleared_after_collective(self):
        from torch.distributed.vnccl import VNCCLProcessGroup

        pgs = self._make_pgs()

        def _work(rank, pg):
            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce(t)
            return t[0]

        self._run_on_pgs(pgs, _work)
        self.assertEqual(len(VNCCLProcessGroup._active), 0)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxModuleFlag(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_run_as_module(self):
        pkg_dir = os.path.join(self._tmpdir, "mypkg")
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(pkg_dir, "train.py"), "w") as f:
            f.write(
                "import os, torch, torch.distributed as dist\n"
                "dist.init_process_group()\n"
                "rank = dist.get_rank()\n"
                "device = f\"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}\"\n"
                "torch.cuda.set_device(device)\n"
                "x = torch.ones(4, device=device)\n"
                "dist.all_reduce(x)\n"
                "dist.destroy_process_group()\n"
            )

        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir)
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = trace_dir
        env["TORCHMUX_NGPUS"] = "1"
        env["PYTHONPATH"] = self._tmpdir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", "-m", "mypkg.train",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux -m failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


class TestTorchmuxArgValidation(TestCase):
    def test_ngpus_greater_than_nproc_rejected(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", "--ngpus", "4", "dummy.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--ngpus", result.stderr)


class TestSharedIntCrossProcess(TestCase):
    def test_cross_process_sharing(self):
        from torch.distributed.torchmux import _SharedInt

        si = _SharedInt()
        self.addCleanup(si.cleanup)
        si.value = 0

        def _child_writer(si_pickled):
            import pickle
            child_si = pickle.loads(si_pickled)
            child_si.value = 42

        import pickle
        import multiprocessing

        p = multiprocessing.Process(
            target=_child_writer,
            args=(pickle.dumps(si),),
        )
        p.start()
        p.join(timeout=10)
        self.assertEqual(p.exitcode, 0)
        self.assertEqual(si.value, 42)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxAVGCorrectness(TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_allreduce_avg(self):
        script = os.path.join(self._tmpdir, "avg_test.py")
        with open(script, "w") as f:
            f.write(
                "import json, os, torch, torch.distributed as dist\n"
                "dist.init_process_group()\n"
                "rank = dist.get_rank()\n"
                "ws = dist.get_world_size()\n"
                "device = f\"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}\"\n"
                "torch.cuda.set_device(device)\n"
                "results = {}\n"
                "x = torch.full((4,), float(rank + 1), device=device)\n"
                "dist.all_reduce(x, op=dist.ReduceOp.AVG)\n"
                "expected = (ws + 1) / 2.0\n"
                "results['allreduce_avg'] = torch.allclose(\n"
                "    x, torch.full_like(x, expected)\n"
                ")\n"
                "full = torch.full((ws * 2,), float(rank + 1), device=device)\n"
                "out = torch.zeros(2, device=device)\n"
                "dist.reduce_scatter_tensor(out, full, op=dist.ReduceOp.AVG)\n"
                "expected_rs = (ws + 1) / 2.0\n"
                "results['reduce_scatter_avg'] = torch.allclose(\n"
                "    out, torch.full_like(out, expected_rs)\n"
                ")\n"
                "dist.destroy_process_group()\n"
                "path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'],\n"
                "                    f'rank{rank}.json')\n"
                "with open(path, 'w') as f:\n"
                "    json.dump({k: bool(v) for k, v in results.items()}, f)\n"
            )

        results_dir = os.path.join(self._tmpdir, "results")
        os.makedirs(results_dir)
        env = os.environ.copy()
        env["TORCHMUX_NGPUS"] = "1"
        env["TORCHMUX_RESULTS_DIR"] = results_dir
        env["TORCHMUX_TRACE_DIR"] = self._tmpdir

        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux AVG test failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

        for rank in range(2):
            rpath = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(rpath))
            with open(rpath) as f:
                results = json.load(f)
            for op_name, passed in results.items():
                self.assertTrue(passed, f"rank {rank}: {op_name} failed")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxCoalescedCollectives(TestCase):
    """Test allgather_into_tensor_coalesced and reduce_scatter_tensor_coalesced
    through the full torchmux subprocess path."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def test_coalesced_correctness(self):
        script = os.path.join(self._tmpdir, "coalesced.py")
        with open(script, "w") as f:
            f.write(
                "import json, os, torch, torch.distributed as dist\n"
                "from torch.distributed import ReduceOp\n"
                "dist.init_process_group()\n"
                "rank = dist.get_rank()\n"
                "ws = dist.get_world_size()\n"
                "device = f\"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}\"\n"
                "torch.cuda.set_device(device)\n"
                "results = {}\n"
                "\n"
                "# allgather_into_tensor_coalesced via _allgather_base path\n"
                "inp = torch.full((2,), float(rank), device=device)\n"
                "out = torch.zeros(2 * ws, device=device)\n"
                "dist.all_gather_into_tensor(out, inp)\n"
                "expected = torch.cat([torch.full((2,), float(r), device=device) for r in range(ws)])\n"
                "results['allgather_base'] = torch.allclose(out, expected)\n"
                "\n"
                "# reduce_scatter_tensor via _reduce_scatter_base path\n"
                "full = torch.full((ws * 2,), float(rank + 1), device=device)\n"
                "out2 = torch.zeros(2, device=device)\n"
                "dist.reduce_scatter_tensor(out2, full)\n"
                "expected_rs = ws * (ws + 1) / 2\n"
                "results['reduce_scatter_base'] = torch.allclose(out2, torch.full_like(out2, expected_rs))\n"
                "\n"
                "dist.destroy_process_group()\n"
                "path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'], f'rank{rank}.json')\n"
                "with open(path, 'w') as f:\n"
                "    json.dump({k: bool(v) for k, v in results.items()}, f)\n"
            )

        results_dir = os.path.join(self._tmpdir, "results")
        os.makedirs(results_dir)
        env = os.environ.copy()
        env["TORCHMUX_NGPUS"] = "1"
        env["TORCHMUX_RESULTS_DIR"] = results_dir
        env["TORCHMUX_TRACE_DIR"] = self._tmpdir

        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.torchmux",
                "--nproc-per-node", "2", script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"torchmux coalesced test failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

        for rank in range(2):
            rpath = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(rpath))
            with open(rpath) as f:
                results = json.load(f)
            for op_name, passed in results.items():
                self.assertTrue(passed, f"rank {rank}: {op_name} failed")


class TestVNCCLSequenceNumbering(_VNCCLTestBase):
    """Verify that the sequence-number fix for _active dict prevents
    collective mismatch when ranks arrive at different speeds."""

    def test_rapid_sequential_collectives(self):
        pgs = self._make_pgs()

        def _work(rank, pg):
            results = []
            for i in range(10):
                t = [torch.full((4,), float(rank + i))]
                pg.allreduce(t)
                results.append(t[0].clone())
            return results

        results = self._run_on_pgs(pgs, _work)
        for i in range(10):
            expected = sum(r + i for r in range(self._world_size))
            for r in range(self._world_size):
                self.assertEqual(
                    results[r][i],
                    torch.full((4,), float(expected)),
                )

    def test_mixed_collective_types(self):
        from torch._C._distributed_c10d import BroadcastOptions

        pgs = self._make_pgs()

        def _work(rank, pg):
            results = []

            t = [torch.full((4,), float(rank + 1))]
            pg.allreduce(t)
            results.append(t[0].clone())

            t = [torch.full((4,), 77.0) if rank == 0 else torch.zeros(4)]
            opts = BroadcastOptions()
            opts.rootRank = 0
            pg.broadcast(t, opts)
            results.append(t[0].clone())

            t = [torch.full((4,), float(rank + 10))]
            pg.allreduce(t)
            results.append(t[0].clone())

            return results

        results = self._run_on_pgs(pgs, _work)
        ws = self._world_size
        expected_ar1 = ws * (ws + 1) / 2
        expected_ar2 = sum(r + 10 for r in range(ws))
        for r in range(ws):
            self.assertEqual(results[r][0], torch.full((4,), expected_ar1))
            self.assertEqual(results[r][1], torch.full((4,), 77.0))
            self.assertEqual(results[r][2], torch.full((4,), float(expected_ar2)))


if __name__ == "__main__":
    run_tests()
