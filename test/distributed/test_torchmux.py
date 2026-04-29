# Owner(s): ["oncall: distributed"]
"""Integration tests for torchmux.

Tests cooperative scheduling trace ordering and collective correctness
under torchmux (store-based PG).
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


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


SINGLE_WORKER_SCRIPT = """
import os
import torch
import torch.distributed as dist

dist.init_process_group()
assert dist.get_rank() == 0
assert dist.get_world_size() == 1
device = f"cuda:{0 % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
x = torch.ones(4, device=device)
dist.all_reduce(x)
assert torch.allclose(x, torch.ones(4, device=device))
dist.destroy_process_group()
"""


CRASH_SCRIPT = """
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
raise RuntimeError('intentional crash')
"""


MODULE_TRAIN_SCRIPT = """
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
x = torch.ones(4, device=device)
dist.all_reduce(x)
dist.destroy_process_group()
"""


AVG_SCRIPT = """
import json
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
results = {}

x = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(x, op=dist.ReduceOp.AVG)
expected = (ws + 1) / 2.0
results['allreduce_avg'] = torch.allclose(x, torch.full_like(x, expected))

full = torch.full((ws * 2,), float(rank + 1), device=device)
out = torch.zeros(2, device=device)
dist.reduce_scatter_tensor(out, full, op=dist.ReduceOp.AVG)
expected_rs = (ws + 1) / 2.0
results['reduce_scatter_avg'] = torch.allclose(out, torch.full_like(out, expected_rs))

dist.destroy_process_group()
path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'], f'rank{rank}.json')
with open(path, 'w') as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


COALESCED_SCRIPT = """
import json
import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
results = {}

# allgather_into_tensor_coalesced via _allgather_base path
inp = torch.full((2,), float(rank), device=device)
out = torch.zeros(2 * ws, device=device)
dist.all_gather_into_tensor(out, inp)
expected = torch.cat([torch.full((2,), float(r), device=device) for r in range(ws)])
results['allgather_base'] = torch.allclose(out, expected)

# reduce_scatter_tensor via _reduce_scatter_base path
full = torch.full((ws * 2,), float(rank + 1), device=device)
out2 = torch.zeros(2, device=device)
dist.reduce_scatter_tensor(out2, full)
expected_rs = ws * (ws + 1) / 2
results['reduce_scatter_base'] = torch.allclose(out2, torch.full_like(out2, expected_rs))

dist.destroy_process_group()
path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'], f'rank{rank}.json')
with open(path, 'w') as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


MULTIGPU_SCRIPT = """
import json
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
ngpus = int(os.environ.get('TORCHMUX_NGPUS', '1'))
device = f'cuda:{rank % ngpus}'
torch.cuda.set_device(device)
results = {}

x = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(x)
expected = ws * (ws + 1) / 2
results['allreduce'] = torch.allclose(x, torch.full_like(x, expected))
results['device_index'] = (torch.cuda.current_device() == rank % ngpus)

dist.destroy_process_group()
path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'], f'rank{rank}.json')
with open(path, 'w') as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


NEW_GROUP_SCRIPT = """
import json
import os
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
ws = dist.get_world_size()
device = f"cuda:{rank % int(os.environ.get('TORCHMUX_NGPUS', '1'))}"
torch.cuda.set_device(device)
results = {}

# Global allreduce to verify the default PG works
x = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(x)
expected_global = ws * (ws + 1) / 2
results['global_allreduce'] = torch.allclose(x, torch.full_like(x, expected_global))

# Create a sub-group with the even ranks
even_ranks = [r for r in range(ws) if r % 2 == 0]
sub_group = dist.new_group(even_ranks)

if rank % 2 == 0:
    y = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(y, group=sub_group)
    expected_sub = sum(r + 1 for r in even_ranks)
    results['sub_allreduce'] = torch.allclose(y, torch.full_like(y, float(expected_sub)))

# Another global allreduce after sub-group usage
z = torch.full((4,), float(rank + 1), device=device)
dist.all_reduce(z)
results['post_subgroup_allreduce'] = torch.allclose(z, torch.full_like(z, expected_global))

dist.destroy_process_group()
path = os.path.join(os.environ['TORCHMUX_RESULTS_DIR'], f'rank{rank}.json')
with open(path, 'w') as f:
    json.dump({k: bool(v) for k, v in results.items()}, f)
"""


class _TorchmuxSubprocessBase(TestCase):
    """Shared base class for torchmux subprocess tests."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, True)

    def _write_script(self, name, content):
        path = os.path.join(self._tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _run_torchmux(self, nproc, script, env_extra=None, ngpus=1, extra_args=None):
        trace_dir = os.path.join(self._tmpdir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        env = os.environ.copy()
        env["TORCHMUX_TRACE_DIR"] = trace_dir
        env["TORCHMUX_NGPUS"] = str(ngpus)
        if env_extra:
            env.update(env_extra)
        cmd = [
            sys.executable, "-m", "torch.distributed.torchmux",
            "--nproc-per-node", str(nproc),
        ]
        if ngpus > 1:
            cmd.extend(["--ngpus", str(ngpus)])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(script)
        return subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=120,
        )

    def _run_and_check_results(self, nproc, script, env_extra=None, ngpus=1):
        results_dir = os.path.join(self._tmpdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        env = {"TORCHMUX_RESULTS_DIR": results_dir}
        if env_extra:
            env.update(env_extra)
        result = self._run_torchmux(nproc, script, env_extra=env, ngpus=ngpus)
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        for rank in range(nproc):
            rpath = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(rpath), f"missing results for rank {rank}")
            with open(rpath) as f:
                results = json.load(f)
            for op_name, passed in results.items():
                self.assertTrue(passed, f"rank {rank}: {op_name} failed")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxTraceOrdering(_TorchmuxSubprocessBase):
    def setUp(self):
        super().setUp()
        self._script = self._write_script("train.py", TRAIN_SCRIPT)

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
        result = self._run_torchmux(3, self._script)
        self.assertEqual(
            result.returncode, 0,
            f"torchmux failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        trace_dir = os.path.join(self._tmpdir, "traces")

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
class TestTorchmuxCollectiveCorrectness(_TorchmuxSubprocessBase):
    def test_collective_correctness_2workers(self):
        script = self._write_script("correctness.py", CORRECTNESS_SCRIPT)
        self._run_and_check_results(2, script)


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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxSingleWorker(_TorchmuxSubprocessBase):
    def test_nproc_1(self):
        script = self._write_script("single.py", SINGLE_WORKER_SCRIPT)
        result = self._run_torchmux(1, script)
        self.assertEqual(
            result.returncode, 0,
            f"torchmux nproc=1 failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxErrorHandling(_TorchmuxSubprocessBase):
    def test_worker_crash_propagates(self):
        script = self._write_script("crash.py", CRASH_SCRIPT)
        result = self._run_torchmux(2, script)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("intentional crash", result.stderr)


class TestTorchmuxModuleFlag(_TorchmuxSubprocessBase):
    def test_run_as_module(self):
        pkg_dir = os.path.join(self._tmpdir, "mypkg")
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(pkg_dir, "train.py"), "w") as f:
            f.write(MODULE_TRAIN_SCRIPT)

        env = {
            "PYTHONPATH": self._tmpdir + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        result = self._run_torchmux(2, "mypkg.train", env_extra=env, extra_args=["-m"])
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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxAVGCorrectness(_TorchmuxSubprocessBase):
    def test_allreduce_avg(self):
        script = self._write_script("avg_test.py", AVG_SCRIPT)
        self._run_and_check_results(2, script)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxCoalescedCollectives(_TorchmuxSubprocessBase):
    """Test allgather_into_tensor_coalesced and reduce_scatter_tensor_coalesced
    through the full torchmux subprocess path."""

    def test_coalesced_correctness(self):
        script = self._write_script("coalesced.py", COALESCED_SCRIPT)
        self._run_and_check_results(2, script)


class TestTorchmuxMultiGPU(_TorchmuxSubprocessBase):
    """Test torchmux with --ngpus 2 to verify multi-GPU device mapping."""

    def test_ngpus_2_correctness(self):
        script = self._write_script("multigpu.py", MULTIGPU_SCRIPT)
        self._run_and_check_results(4, script, ngpus=2)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestTorchmuxNewGroup(_TorchmuxSubprocessBase):
    """Test that dist.new_group() works through torchmux's monkey-patch."""

    def test_new_group_with_subgroup_collective(self):
        script = self._write_script("new_group.py", NEW_GROUP_SCRIPT)
        self._run_and_check_results(4, script)


if __name__ == "__main__":
    run_tests()
