# Owner(s): ["oncall: distributed"]
#
# Correctness tests for the TCCL c10d backend (RDMA-over-Thunderbolt for Apple
# Silicon MPS tensors).
#
# LAUNCH MODEL — read this before adding tests.
# TCCL is a genuinely MULTI-NODE backend: it moves data over RDMA between
# *physical* Macs linked by Thunderbolt, so — unlike Gloo/NCCL, which exercise a
# backend with multiple processes (or GPUs) on a single host — it cannot be
# driven by the single-host spawn model of MultiProcessTestCase /
# MultiProcContinuousTest (both spawn local processes + rendezvous via a shared
# FileStore, neither of which crosses machines). Instead these tests are
# ENV-LAUNCHED: one rank per machine, each process runs this file with
# RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT (and the per-rank
# TCCL_PEER_DEVICES the backend reads) provided by the TCCL multi-node launcher
# (scripts/tccl_launch.py). Rendezvous is a TCPStore over the mesh; the
# collectives run in lockstep because every rank executes the identical,
# deterministically-ordered set of test methods.
#
# Consequence: a plain ``python test_c10d_tccl.py`` or pytest invocation on a
# single host (no launcher env) SKIPS the whole suite — TCCL needs the fabric.
# This is the same "collected but hardware-gated, skips without the hardware"
# property NCCL tests have for GPUs; how CI should drive a multi-host backend is
# an open item for the upstream RFC.
#
# To run on a configured cluster (rank 0 = master), e.g.:
#   python scripts/tccl_launch.py --hostfile ~/tccl_hostfile.json --nodes 4 \
#       --port 29500 -- test_c10d_tccl.py
# (the launcher sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT/TCCL_PEER_DEVICES
# per rank and runs this file on each node).

import os
import contextlib
import math
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as c10d
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# TCCL needs the multi-node launcher env. Without it (e.g. generic CI / a bare
# pytest run) there is no fabric to talk to, so skip the module cleanly.

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import requires_tccl
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


# ---- dtype / reduce-op tables under test ----
REDUCE_DTYPES = [
    torch.float32, torch.float16, torch.bfloat16,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
]
COPY_DTYPES = REDUCE_DTYPES + [torch.complex64]
FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _fill_value(rank, dtype):
    if dtype == torch.bool:
        return bool(rank % 2 == 0)
    if dtype == torch.complex64:
        return complex(rank + 1, rank + 1)
    return rank + 1


def _tol(dtype):
    # tiny integer sums are exact in fp16/bf16 too; keep a small cushion for them.
    if dtype in (torch.float16, torch.bfloat16):
        return dict(rtol=1e-2, atol=1e-2)
    return dict(rtol=0, atol=0)


def _launcher_env():
    """(rank, world_size, master_addr, master_port) from the launcher, or None.

    Read from the explicit CLI args the launcher passes (--rank/--world-size/
    --master-addr/--port), falling back to the RANK/WORLD_SIZE/MASTER_ADDR/
    MASTER_PORT env vars. The CLI args are the source of truth: a cluster login
    shell can carry a stale MASTER_PORT in the environment (a duplicate that
    resolves to the wrong port for ssh-launched ranks), which would desync
    rank 0 (launched locally, no login shell) from the other ranks."""
    import argparse

    def _ei(k):
        v = os.environ.get(k)
        return int(v) if v is not None else None

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rank", type=int)
    p.add_argument("--world-size", type=int)
    p.add_argument("--master-addr")
    p.add_argument("--port", type=int)
    a, _ = p.parse_known_args()
    rank = a.rank if a.rank is not None else _ei("RANK")
    ws = a.world_size if a.world_size is not None else _ei("WORLD_SIZE")
    addr = a.master_addr or os.environ.get("MASTER_ADDR")
    port = a.port if a.port is not None else _ei("MASTER_PORT")
    if rank is None or ws is None or addr is None or port is None:
        return None
    return (rank, ws, addr, port)


# Captured at import — BEFORE the __main__ block strips these flags from argv.
_LAUNCH = _launcher_env()


# Integration-smoke model dims + a seeded RNG helper (CPU-generated then moved to
# MPS so every rank produces identical/deterministic tensors from a given seed).
_IN, _HID, _OUT, _BATCH = 64, 128, 64, 8


def _randn(shape, seed, dtype, device):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype).to(device)


def _ring_topology():
    """True when the launcher brought the group up as a ring (sparse cabling):
    TCCL_TOPOLOGY=ring. Ring drops the two mesh-only capabilities (uneven
    alltoall, non-neighbor p2p), so those tests invert to rejection tests."""
    return os.environ.get("TCCL_TOPOLOGY", "mesh").lower() == "ring"


@contextlib.contextmanager
def _force_algo(algo):
    """Force the TCCL ring/mesh algorithm for the enclosed collective(s). Every
    rank runs the identical parametrized test, so all ranks set the same algo (a
    ring/mesh mix across ranks would deadlock). At world_size <= 2 ring silently
    falls back to mesh (still correct)."""
    prev = os.environ.get("TCCL_FORCE_ALGO")
    os.environ["TCCL_FORCE_ALGO"] = algo
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TCCL_FORCE_ALGO", None)
        else:
            os.environ["TCCL_FORCE_ALGO"] = prev


class TcclTestBase(TestCase):
    """Env-launched base: one rank per machine. Inits a single TCCL process group
    for the whole class (rendezvous via TCPStore from the launcher env) and tears
    it down at the end. A barrier between tests keeps the ranks in lockstep."""

    timeout = timedelta(seconds=120)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        env = _LAUNCH
        if env is None:
            raise unittest.SkipTest(
                "TCCL is multi-node; launch via the TCCL launcher (sets RANK/"
                "WORLD_SIZE/MASTER_ADDR/MASTER_PORT). Skipping single-host run.")
        if sys.platform != "darwin" or not c10d.is_tccl_available():
            raise unittest.SkipTest("TCCL requires macOS with the TCCL backend built in")
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("TCCL requires an available MPS device")

        cls.rank, cls.world_size, master_addr, master_port = env
        cls.device = torch.device("mps", 0)  # one MPS device per machine (not mps:rank)
        store = c10d.TCPStore(
            host_name=master_addr, port=master_port, world_size=cls.world_size,
            is_master=(cls.rank == 0), timeout=cls.timeout)
        c10d.init_process_group(
            backend="tccl", store=store, rank=cls.rank, world_size=cls.world_size,
            device_id=cls.device, timeout=cls.timeout)

    @classmethod
    def tearDownClass(cls):
        if c10d.is_initialized():
            c10d.destroy_process_group()
        super().tearDownClass()

    def tearDown(self):
        # Resync ranks between tests so a per-test mismatch can't desync the next
        # (mirrors the inter-cell barrier in the external dtype-matrix harness).
        if c10d.is_initialized():
            c10d.barrier()
        super().tearDown()

    # ---- helpers --------------------------------------------------------------
    def _full(self, dtype, n=4096):
        return torch.full((n,), _fill_value(self.rank, dtype), dtype=dtype,
                          device=self.device)

    def _assert(self, actual, expected_scalar, dtype):
        exp = torch.full_like(actual, expected_scalar)
        self.assertEqual(actual, exp, **_tol(dtype))


@requires_tccl()
class ProcessGroupTcclCorrectnessTest(TcclTestBase):
    """Per-collective numerical correctness over the supported dtype/op contract."""

    # ---- allreduce ------------------------------------------------------------
    @parametrize("dtype", REDUCE_DTYPES)
    def test_allreduce_sum(self, dtype):
        t = self._full(dtype)
        c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
        ws = self.world_size
        if dtype == torch.bool:
            self._assert(t, True, dtype)  # SUM == OR; any rank0 True -> True
        else:
            self._assert(t, ws * (ws + 1) // 2, dtype)

    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("op", ["MIN", "MAX"])
    def test_allreduce_minmax(self, dtype, op):
        t = self._full(dtype)
        c10d.all_reduce(t, op=getattr(c10d.ReduceOp, op))
        if dtype == torch.bool:
            self._assert(t, (op == "MAX"), dtype)  # MIN=AND=False, MAX=OR=True
        else:
            self._assert(t, 1 if op == "MIN" else self.world_size, dtype)

    @parametrize("dtype", FLOAT_DTYPES)
    def test_allreduce_avg(self, dtype):
        t = self._full(dtype)
        c10d.all_reduce(t, op=c10d.ReduceOp.AVG)
        self._assert(t, (self.world_size + 1) / 2.0, dtype)

    @parametrize("dtype", REDUCE_DTYPES)
    def test_allreduce_product(self, dtype):
        t = self._full(dtype)
        c10d.all_reduce(t, op=c10d.ReduceOp.PRODUCT)
        if dtype == torch.bool:
            self._assert(t, False, dtype)  # PRODUCT == AND; even ranks True, odd False
        else:
            # prod over ranks of (rank+1) = world_size! (fits int8 for ws <= 5)
            self._assert(t, math.factorial(self.world_size), dtype)

    # ---- broadcast (byte-copy, any dtype) -------------------------------------
    @parametrize("dtype", COPY_DTYPES)
    def test_broadcast(self, dtype):
        t = self._full(dtype)
        c10d.broadcast(t, src=0)
        self._assert(t, _fill_value(0, dtype), dtype)

    # ---- allgather (byte-copy) ------------------------------------------------
    @parametrize("dtype", COPY_DTYPES)
    def test_allgather_base(self, dtype):
        ws = self.world_size
        inp = self._full(dtype, n=1024)
        out = torch.empty(1024 * ws, dtype=dtype, device=self.device)
        c10d.all_gather_into_tensor(out, inp)
        for r in range(ws):
            exp = torch.full((1024,), _fill_value(r, dtype), dtype=dtype, device=self.device)
            self.assertEqual(out[r * 1024:(r + 1) * 1024], exp, **_tol(dtype))

    # ---- reduce_scatter (coalesced; TP sequence-parallel / DTensor path) ------
    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("op", ["SUM", "MIN", "MAX"])
    def test_reduce_scatter_sum_minmax(self, dtype, op):
        import torch.distributed._functional_collectives as funcol
        ws = self.world_size
        inp = self._full(dtype, n=1024 * ws)
        out = funcol.reduce_scatter_tensor(inp, op.lower(), scatter_dim=0,
                                           group=c10d.distributed_c10d._get_default_group())
        out = out.wait() if hasattr(out, "wait") else out
        if dtype == torch.bool:
            want = True if op in ("SUM", "MAX") else False
            self._assert(out, want, dtype)
        else:
            want = {"SUM": ws * (ws + 1) // 2, "MIN": 1, "MAX": ws}[op]
            self._assert(out, want, dtype)

    @parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_scatter_avg(self, dtype):
        import torch.distributed._functional_collectives as funcol
        ws = self.world_size
        inp = self._full(dtype, n=1024 * ws)
        out = funcol.reduce_scatter_tensor(inp, "avg", scatter_dim=0,
                                           group=c10d.distributed_c10d._get_default_group())
        out = out.wait() if hasattr(out, "wait") else out
        self._assert(out, (ws + 1) / 2.0, dtype)

    @parametrize("dtype", REDUCE_DTYPES)
    def test_reduce_scatter_product(self, dtype):
        import torch.distributed._functional_collectives as funcol
        ws = self.world_size
        inp = self._full(dtype, n=1024 * ws)
        out = funcol.reduce_scatter_tensor(inp, "product", scatter_dim=0,
                                           group=c10d.distributed_c10d._get_default_group())
        out = out.wait() if hasattr(out, "wait") else out
        if dtype == torch.bool:
            self._assert(out, False, dtype)
        else:
            self._assert(out, math.factorial(ws), dtype)

    # ---- barrier --------------------------------------------------------------
    def test_barrier(self):
        c10d.barrier()  # no data; should not raise

    # ---- reduce (rooted reduction) --------------------------------------------
    # Reduce lands on every rank (all-reduce), but only the root is guaranteed -
    # assert there. fill -> SUM = ws(ws+1)/2, MAX = ws.
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
    @parametrize("op", ["SUM", "MAX"])
    def test_reduce_to_root(self, dtype, op):
        root = self.world_size - 1
        t = self._full(dtype)
        c10d.reduce(t, dst=root, op=getattr(c10d.ReduceOp, op))
        if self.rank == root:
            if dtype == torch.bool:
                # SUM/MAX = OR -> True
                self._assert(t, True, dtype)
            else:
                ws = self.world_size
                self._assert(t, ws * (ws + 1) // 2 if op == "SUM" else ws, dtype)

    @parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_avg(self, dtype):
        t = self._full(dtype)
        c10d.reduce(t, dst=0, op=c10d.ReduceOp.AVG)
        if self.rank == 0:
            self._assert(t, (self.world_size + 1) / 2.0, dtype)

    @parametrize("dtype", [torch.float32, torch.int64])
    def test_reduce_product(self, dtype):
        root = self.world_size - 1
        t = self._full(dtype)
        c10d.reduce(t, dst=root, op=c10d.ReduceOp.PRODUCT)
        if self.rank == root:
            expected = 1
            for k in range(1, self.world_size + 1):
                expected *= k
            self._assert(t, expected, dtype)

    # ---- gather (root collects every rank's tensor) ---------------------------
    # Byte-copy -> any dtype. Root slot k == fill(k). Mesh-only; ring rejects it.
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.complex64])
    def test_gather(self, dtype):
        if _ring_topology():
            self.skipTest("gather is mesh-only; ring rejects it "
                          "(see ProcessGroupTcclContractTest)")
        ws, r, dev, n = self.world_size, self.rank, self.device, 1024
        inp = self._full(dtype, n=n)
        if r == 0:
            out = [torch.empty(n, dtype=dtype, device=dev) for _ in range(ws)]
            c10d.gather(inp, gather_list=out, dst=0)
            for k in range(ws):
                exp = torch.full((n,), _fill_value(k, dtype), dtype=dtype, device=dev)
                self.assertEqual(out[k], exp, **_tol(dtype))
        else:
            c10d.gather(inp, gather_list=None, dst=0)

    def test_gather_nonzero_root(self):
        if _ring_topology():
            self.skipTest("gather is mesh-only; ring rejects it")
        ws, r, dev, n = self.world_size, self.rank, self.device, 1024
        root = ws - 1
        inp = self._full(torch.float32, n=n)
        if r == root:
            out = [torch.empty(n, dtype=torch.float32, device=dev) for _ in range(ws)]
            c10d.gather(inp, gather_list=out, dst=root)
            for k in range(ws):
                self.assertEqual(out[k], torch.full((n,), float(k + 1), device=dev))
        else:
            c10d.gather(inp, gather_list=None, dst=root)

    # ---- scatter (root distributes a per-rank slice) --------------------------
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.complex64])
    def test_scatter(self, dtype):
        if _ring_topology():
            self.skipTest("scatter is mesh-only; ring rejects it "
                          "(see ProcessGroupTcclContractTest)")
        ws, r, dev, n = self.world_size, self.rank, self.device, 1024
        out = torch.empty(n, dtype=dtype, device=dev)
        if r == 0:
            sl = [torch.full((n,), _fill_value(k, dtype), dtype=dtype, device=dev)
                  for k in range(ws)]
            c10d.scatter(out, scatter_list=sl, src=0)
        else:
            c10d.scatter(out, scatter_list=None, src=0)
        self._assert(out, _fill_value(r, dtype), dtype)

    def test_scatter_nonzero_root(self):
        if _ring_topology():
            self.skipTest("scatter is mesh-only; ring rejects it")
        ws, r, dev, n = self.world_size, self.rank, self.device, 1024
        root = ws - 1
        out = torch.empty(n, dtype=torch.float32, device=dev)
        if r == root:
            sl = [torch.full((n,), float(k + 1), dtype=torch.float32, device=dev)
                  for k in range(ws)]
            c10d.scatter(out, scatter_list=sl, src=root)
        else:
            c10d.scatter(out, scatter_list=None, src=root)
        self.assertEqual(out, torch.full((n,), float(r + 1), device=dev))

    # ---- reduce_scatter, tensor form (FSDP reduce_scatter_tensor ->
    #      _reduce_scatter_base). Forces BOTH ring and mesh; world_size <= 2 makes
    #      ring fall back to mesh (still correct). Same fill -> same expected value
    #      as the coalesced path: out[k] = reduction over ranks of (rank+1).
    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("op", ["SUM", "MIN", "MAX"])
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_base_sum_minmax(self, dtype, op, algo):
        with _force_algo(algo):
            ws = self.world_size
            inp = self._full(dtype, n=1024 * ws)
            out = torch.empty(1024, dtype=dtype, device=self.device)
            c10d.reduce_scatter_tensor(out, inp, op=getattr(c10d.ReduceOp, op))
            if dtype == torch.bool:
                self._assert(out, op in ("SUM", "MAX"), dtype)  # SUM/MAX=OR, MIN=AND
            else:
                self._assert(out, {"SUM": ws * (ws + 1) // 2, "MIN": 1, "MAX": ws}[op], dtype)

    @parametrize("dtype", FLOAT_DTYPES)
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_base_avg(self, dtype, algo):
        with _force_algo(algo):
            ws = self.world_size
            inp = self._full(dtype, n=1024 * ws)
            out = torch.empty(1024, dtype=dtype, device=self.device)
            c10d.reduce_scatter_tensor(out, inp, op=c10d.ReduceOp.AVG)
            self._assert(out, (ws + 1) / 2.0, dtype)

    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_base_product(self, dtype, algo):
        with _force_algo(algo):
            ws = self.world_size
            inp = self._full(dtype, n=1024 * ws)
            out = torch.empty(1024, dtype=dtype, device=self.device)
            c10d.reduce_scatter_tensor(out, inp, op=c10d.ReduceOp.PRODUCT)
            if dtype == torch.bool:
                self._assert(out, False, dtype)
            else:
                self._assert(out, math.factorial(ws), dtype)

    # ---- reduce_scatter, list form (dist.reduce_scatter -> the `reduce_scatter`
    #      virtual). out (on rank r) = reduction over ranks of in_list[r]; every
    #      rank fills each in_list[j] = rank+1, so out = reduction over ranks of
    #      (rank+1) — same expected value. Exercises the zero-copy per-rank-pointer
    #      path. Both ring and mesh.
    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("op", ["SUM", "MIN", "MAX"])
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_list(self, dtype, op, algo):
        with _force_algo(algo):
            ws = self.world_size
            out = torch.empty(1024, dtype=dtype, device=self.device)
            in_list = [self._full(dtype, n=1024) for _ in range(ws)]
            c10d.reduce_scatter(out, in_list, op=getattr(c10d.ReduceOp, op))
            if dtype == torch.bool:
                self._assert(out, op in ("SUM", "MAX"), dtype)
            else:
                self._assert(out, {"SUM": ws * (ws + 1) // 2, "MIN": 1, "MAX": ws}[op], dtype)

    @parametrize("dtype", FLOAT_DTYPES)
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_list_avg(self, dtype, algo):
        with _force_algo(algo):
            ws = self.world_size
            out = torch.empty(1024, dtype=dtype, device=self.device)
            in_list = [self._full(dtype, n=1024) for _ in range(ws)]
            c10d.reduce_scatter(out, in_list, op=c10d.ReduceOp.AVG)
            self._assert(out, (ws + 1) / 2.0, dtype)

    @parametrize("dtype", REDUCE_DTYPES)
    @parametrize("algo", ["ring", "mesh"])
    def test_reduce_scatter_list_product(self, dtype, algo):
        with _force_algo(algo):
            ws = self.world_size
            out = torch.empty(1024, dtype=dtype, device=self.device)
            in_list = [self._full(dtype, n=1024) for _ in range(ws)]
            c10d.reduce_scatter(out, in_list, op=c10d.ReduceOp.PRODUCT)
            if dtype == torch.bool:
                self._assert(out, False, dtype)
            else:
                self._assert(out, math.factorial(ws), dtype)

    # ---- send / recv (point-to-point over the per-peer UC QP) -----------------
    # Byte-copy, any dtype. TCCL's single worker thread serializes ops, so a rank
    # never has a send and a recv in flight at once; these use directed/sequential
    # patterns. Concurrent bidirectional P2P (e.g. naive batch_isend_irecv both
    # ways) would serialize on the worker — a pipeline-framework concern, out of
    # scope here. No recv-before-send handshake: the ctor RTS barrier brings the
    # QPs up and the sender's stage-copy gives the receiver a head start to post
    # its recv.
    def test_send_recv_chain(self):
        # Sequential pipeline 0->1->...->(ws-1): rank i sends float(i) to i+1;
        # rank i+1 verifies it got float(i). No cycle -> deadlock-free, any ws>=2.
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 2:
            self.skipTest("send/recv needs world_size >= 2")
        n = 2048
        if r > 0:
            t = torch.empty(n, dtype=torch.float32, device=dev)
            c10d.recv(t, src=r - 1)
            self.assertEqual(t, torch.full((n,), float(r - 1), device=dev))
        if r < ws - 1:
            c10d.send(torch.full((n,), float(r), dtype=torch.float32, device=dev), dst=r + 1)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
    def test_send_recv_pair_dtypes(self, dtype):
        # Directed rank 0 -> rank 1 (other ranks idle). Byte-copy across dtypes.
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 2:
            self.skipTest("send/recv needs world_size >= 2")
        n = 512
        if r == 0:
            c10d.send(torch.full((n,), _fill_value(7, dtype), dtype=dtype, device=dev), dst=1)
        elif r == 1:
            t = torch.empty(n, dtype=dtype, device=dev)
            c10d.recv(t, src=0)
            self._assert(t, _fill_value(7, dtype), dtype)

    def test_send_recv_large_multichunk(self):
        # 1 MB fp32 (> the 512 KB chunk) rank 0 -> rank 1: exercises the chunk loop.
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 2:
            self.skipTest("send/recv needs world_size >= 2")
        n = 256 * 1024  # 1 MB fp32 -> 2 chunks
        if r == 0:
            c10d.send(torch.full((n,), 3.0, dtype=torch.float32, device=dev), dst=1)
        elif r == 1:
            t = torch.empty(n, dtype=torch.float32, device=dev)
            c10d.recv(t, src=0)
            self.assertEqual(t, torch.full((n,), 3.0, device=dev))

    def test_send_recv_nonadjacent(self):
        # rank 0 -> rank (ws-1) directly: any pair has a usable QP (full mesh),
        # not just neighbours.
        if _ring_topology():
            self.skipTest("non-neighbor p2p is mesh-only; ring rejects it "
                          "(see ProcessGroupTcclRingRejectTest)")
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 3:
            self.skipTest("non-adjacent needs world_size >= 3")
        n = 1024
        if r == 0:
            c10d.send(torch.full((n,), 5.0, dtype=torch.float32, device=dev), dst=ws - 1)
        elif r == ws - 1:
            t = torch.empty(n, dtype=torch.float32, device=dev)
            c10d.recv(t, src=0)
            self.assertEqual(t, torch.full((n,), 5.0, device=dev))

    # ---- alltoall_base (MoE / sequence-parallel) ------------------------------
    # Equal split: rank r fills its whole input with fill(r); each output block i
    # (received from rank i) must be fill(i). Byte-copy -> a few dtypes; both ring
    # and mesh (ring needs ws>2, else falls back to mesh — still correct).
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
    @parametrize("algo", ["ring", "mesh"])
    def test_alltoall_equal(self, dtype, algo):
        with _force_algo(algo):
            ws, r, dev = self.world_size, self.rank, self.device
            seg = 256
            inp = torch.full((seg * ws,), _fill_value(r, dtype), dtype=dtype, device=dev)
            out = torch.empty(seg * ws, dtype=dtype, device=dev)
            c10d.all_to_all_single(out, inp)
            for i in range(ws):
                exp = torch.full((seg,), _fill_value(i, dtype), dtype=dtype, device=dev)
                self.assertEqual(out[i * seg:(i + 1) * seg], exp, **_tol(dtype))

    def test_alltoall_uneven(self):
        # alltoallv (variable splits -> mesh). rank r sends (j+1) rows to rank j and
        # fills its input with (r+1); so rank r receives (r+1) rows from each src,
        # each block valued (src+1). Splits are consistent across ranks.
        if _ring_topology():
            self.skipTest("uneven alltoall is mesh-only; ring rejects it "
                          "(see ProcessGroupTcclRingRejectTest)")
        ws, r, dev = self.world_size, self.rank, self.device
        in_splits = [j + 1 for j in range(ws)]
        out_splits = [r + 1 for _ in range(ws)]
        inp = torch.full((sum(in_splits),), float(r + 1), dtype=torch.float32, device=dev)
        out = torch.empty(sum(out_splits), dtype=torch.float32, device=dev)
        c10d.all_to_all_single(out, inp, out_splits, in_splits)
        off = 0
        for src in range(ws):
            self.assertEqual(out[off:off + (r + 1)],
                             torch.full((r + 1,), float(src + 1), device=dev))
            off += r + 1

    @parametrize("algo", ["ring", "mesh"])
    def test_alltoall_large_multichunk(self, algo):
        # Equal split, 1 MB fp32 per segment (> the 512 KB chunk) -> chunk loop.
        with _force_algo(algo):
            ws, r, dev = self.world_size, self.rank, self.device
            seg = 256 * 1024
            inp = torch.full((seg * ws,), float(r + 1), dtype=torch.float32, device=dev)
            out = torch.empty(seg * ws, dtype=torch.float32, device=dev)
            c10d.all_to_all_single(out, inp)
            for i in range(ws):
                self.assertEqual(out[i * seg:(i + 1) * seg],
                                 torch.full((seg,), float(i + 1), device=dev))

    # ---- allgather, list form (dist.all_gather -> the `allgather` virtual) -----
    # The byte-copy paths are dtype-agnostic (proven across dtypes via the _base
    # form above); here we just exercise the separate list-form code path DDP
    # construction uses. fp32 is representative.
    def test_allgather_list(self):
        ws = self.world_size
        inp = self._full(torch.float32, n=1024)
        outs = [torch.empty(1024, dtype=torch.float32, device=self.device) for _ in range(ws)]
        c10d.all_gather(outs, inp)
        for r in range(ws):
            self.assertEqual(outs[r], torch.full((1024,), float(r + 1), device=self.device))

    # ---- allgather coalesced (funcol -> allgather_into_tensor_coalesced) -------
    # The DTensor/TP entry point, distinct from _base (eager) above.
    def test_allgather_coalesced(self):
        import torch.distributed._functional_collectives as funcol
        ws = self.world_size
        inp = self._full(torch.float32, n=1024)
        out = funcol.all_gather_tensor(inp, gather_dim=0,
                                       group=c10d.distributed_c10d._get_default_group())
        out = out.wait() if hasattr(out, "wait") else out
        for r in range(ws):
            self.assertEqual(out[r * 1024:(r + 1) * 1024],
                             torch.full((1024,), float(r + 1), device=self.device))

    # ---- large allreduce: crosses the 512 KB mesh chunk boundary + (ws>2) the ring
    #      auto-switch (fp32 >=1 MB). Tiny dtype-matrix cells never reach either. -----
    def test_allreduce_large_fp32(self):
        n = 3 * 1024 * 1024  # 12 MB fp32 -> ring at ws>2, many 512 KB chunks
        t = torch.full((n,), float(self.rank + 1), dtype=torch.float32, device=self.device)
        c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
        ws = self.world_size
        self.assertEqual(t, torch.full((n,), float(ws * (ws + 1) // 2), device=self.device))

    # ---- ragged / chunk-boundary sizes on the RING path ----
    # N full 512 KB chunks + a 1-elem ODD tail: exercises the double-ring A/B
    # half-split and the ragged last chunk. Ring is FORCED (ws<=2 falls back to
    # mesh, still correct). fill = rank+1 -> SUM = ws(ws+1)/2, MAX = ws.
    # 128*1024 fp32 elems = one 512 KB chunk.
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_allreduce_ring_ragged(self, dtype):
        # bf16 also drives the native __bf16 reduce.
        with _force_algo("ring"):
            n = 6 * 128 * 1024 + 1
            t = torch.full((n,), float(self.rank + 1), dtype=dtype, device=self.device)
            c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
            self._assert(t, self.world_size * (self.world_size + 1) // 2, dtype)

    def test_allreduce_ring_chunk_plus_one(self):
        # Exactly one full 512 KB chunk + a 1-element second chunk: the depth-2
        # pipeline must fill/drain a piece far smaller than kChunkSize.
        with _force_algo("ring"):
            n = 128 * 1024 + 1
            t = torch.full((n,), float(self.rank + 1), dtype=torch.float32, device=self.device)
            c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
            ws = self.world_size
            self.assertEqual(t, torch.full((n,), float(ws * (ws + 1) // 2), device=self.device))

    @parametrize("op", ["SUM", "MAX"])
    def test_reduce_scatter_ring_ragged(self, op):
        # bidir ring_reduce_scatter with an odd, non-chunk-aligned per-rank chunk.
        with _force_algo("ring"):
            ws = self.world_size
            n = 3 * 128 * 1024 + 1
            inp = torch.full((n * ws,), float(self.rank + 1), dtype=torch.float32, device=self.device)
            out = torch.empty(n, dtype=torch.float32, device=self.device)
            c10d.reduce_scatter_tensor(out, inp, op=getattr(c10d.ReduceOp, op))
            want = ws * (ws + 1) // 2 if op == "SUM" else ws
            self.assertEqual(out, torch.full((n,), float(want), device=self.device))

    def test_allgather_ring_ragged(self):
        # bidir ring_all_gather with an odd, non-chunk-aligned per-rank shard.
        with _force_algo("ring"):
            ws = self.world_size
            n = 3 * 128 * 1024 + 1
            inp = torch.full((n,), float(self.rank + 1), dtype=torch.float32, device=self.device)
            out = torch.empty(n * ws, dtype=torch.float32, device=self.device)
            c10d.all_gather_into_tensor(out, inp)
            for r in range(ws):
                self.assertEqual(out[r * n:(r + 1) * n],
                                 torch.full((n,), float(r + 1), device=self.device))

    def test_send_recv_ragged_multichunk(self):
        # depth-2 pipelined p2p over an odd, non-chunk-aligned payload (rank 0 -> 1).
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 2:
            self.skipTest("send/recv needs world_size >= 2")
        n = 3 * 128 * 1024 + 1
        if r == 0:
            c10d.send(torch.full((n,), 3.0, dtype=torch.float32, device=dev), dst=1)
        elif r == 1:
            t = torch.empty(n, dtype=torch.float32, device=dev)
            c10d.recv(t, src=0)
            self.assertEqual(t, torch.full((n,), 3.0, device=dev))

    def test_ring_mesh_agree_ragged(self):
        # Cross-check: ragged all_reduce under ring and mesh must agree and equal
        # the analytic SUM - guards the bidir-ring result against the mesh path.
        ws, n = self.world_size, 3 * 128 * 1024 + 1
        exp = torch.full((n,), float(ws * (ws + 1) // 2), device=self.device)
        for algo in ("ring", "mesh"):
            with _force_algo(algo):
                t = torch.full((n,), float(self.rank + 1), dtype=torch.float32, device=self.device)
                c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
                self.assertEqual(t, exp, msg=f"algo={algo}")

    # ---- TCCL_FORCE_ALGO knob: both ring and mesh produce correct results ------
    # (correctness holds regardless of the algo actually chosen; at ws<=2 ring
    # silently falls back to mesh, which is still correct).
    def test_force_algo(self):
        ws = self.world_size
        prev = os.environ.get("TCCL_FORCE_ALGO")
        try:
            for algo, dtype in (("ring", torch.float32), ("mesh", torch.bfloat16)):
                os.environ["TCCL_FORCE_ALGO"] = algo
                t = self._full(dtype)
                c10d.all_reduce(t, op=c10d.ReduceOp.SUM)
                self._assert(t, ws * (ws + 1) // 2, dtype)
        finally:
            if prev is None:
                os.environ.pop("TCCL_FORCE_ALGO", None)
            else:
                os.environ["TCCL_FORCE_ALGO"] = prev

    # ---- process-group identity -----------------------------------------------
    def test_rank_world_size(self):
        self.assertEqual(c10d.get_rank(), self.rank)
        self.assertEqual(c10d.get_world_size(), self.world_size)


@requires_tccl()
class ProcessGroupTcclIntegrationTest(TcclTestBase):
    """End-to-end smokes: real DDP backward and Megatron tensor-parallel forward
    over TCCL, each diffed against a LOCAL single-process baseline (non-circular —
    the baseline uses no collectives)."""

    def _mlp(self):
        return nn.Sequential(nn.Linear(_IN, _HID), nn.ReLU(), nn.Linear(_HID, _OUT))

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_ddp_backward(self, dtype):
        # DDP all-reduce-averages per-rank grads; for a mean-MSE loss with equal
        # per-rank batches that equals the single-process grad over the concatenated
        # batch (computed locally from the same init weights). Exercises DDP ctor
        # (broadcast params + allgather metadata) + backward allreduce over TCCL.
        rank, ws = self.rank, self.world_size
        dev = self.device
        torch.manual_seed(0)                       # identical init on every rank
        model = self._mlp().to(dev).to(dtype)
        init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

        ddp = DDP(model)
        x = _randn((_BATCH, _IN), 100 + rank, dtype, dev)
        y = _randn((_BATCH, _OUT), 200 + rank, dtype, dev)
        ((ddp(x) - y) ** 2).mean().backward()
        torch.mps.synchronize()
        ddp_grads = [p.grad.detach().float().cpu() for p in model.parameters()]

        ref = self._mlp().to(dev).to(dtype)
        ref.load_state_dict(init_sd)
        xs = torch.cat([_randn((_BATCH, _IN), 100 + r, dtype, dev) for r in range(ws)], 0)
        ys = torch.cat([_randn((_BATCH, _OUT), 200 + r, dtype, dev) for r in range(ws)], 0)
        ((ref(xs) - ys) ** 2).mean().backward()
        torch.mps.synchronize()
        ref_grads = [p.grad.detach().float().cpu() for p in ref.parameters()]

        worst = max((dg - rg).abs().max().item() for dg, rg in zip(ddp_grads, ref_grads))
        tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
        self.assertLess(worst, tol, f"DDP grads diverge from single-process baseline: {worst:.3e}")

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_tp_megatron_forward(self, dtype):
        # Megatron MLP: fc1 column-parallel (no comm), fc2 row-parallel
        # (all_reduce-SUM of partials). Reduced output == full single-process MLP.
        rank, ws = self.rank, self.world_size
        dev = self.device
        if _HID % ws != 0:
            self.skipTest(f"hidden dim {_HID} not divisible by world_size {ws}")
        W1 = _randn((_IN, _HID), 1, dtype, dev)    # full weights, identical per rank
        W2 = _randn((_HID, _OUT), 2, dtype, dev)
        x = _randn((_BATCH, _IN), 3, dtype, dev)
        full = torch.relu(x @ W1) @ W2             # single-process reference (no comm)

        hp = _HID // ws
        z = torch.relu(x @ W1[:, rank * hp:(rank + 1) * hp]) @ W2[rank * hp:(rank + 1) * hp, :]
        c10d.all_reduce(z, op=c10d.ReduceOp.SUM)   # sum partials -> full output
        torch.mps.synchronize()

        ref = full.float().cpu()
        abs_err = (z.float().cpu() - ref).abs().max().item()
        # TP activations are large-magnitude; one bf16 ulp at that scale ~1.0, so
        # compare RELATIVE error against the reference scale.
        scale = max(1.0, ref.abs().max().item())
        rtol = 3e-2 if dtype == torch.bfloat16 else 1e-5
        self.assertLess(abs_err / scale, rtol,
                        f"TP forward diverges from single-process baseline: rel={abs_err/scale:.3e}")


@requires_tccl()
class ProcessGroupTcclContractTest(TcclTestBase):
    """Negative/contract tests: the unsupported (op, dtype) cells must reject
    cleanly (raise), not crash or silently corrupt."""

    def test_avg_integer_rejects(self):
        # AVG is float-only (matches NCCL); integer AVG must raise.
        t = self._full(torch.int32)
        with self.assertRaises(Exception):
            c10d.all_reduce(t, op=c10d.ReduceOp.AVG)

    def test_uint16_reduce_rejects(self):
        # uint16/32/64 have no MPS add kernel -> reduce must reject.
        t = self._full(torch.uint16)
        with self.assertRaises(Exception):
            c10d.all_reduce(t, op=c10d.ReduceOp.SUM)

    def test_complex_minmax_rejects(self):
        # complex MIN/MAX is undefined; the PyTorch frontend deny-lists it.
        t = torch.full((128,), complex(self.rank + 1, 1), dtype=torch.complex64,
                       device=self.device)
        with self.assertRaises(Exception):
            c10d.all_reduce(t, op=c10d.ReduceOp.MAX)

    def test_complex_product_rejects(self):
        # complex PRODUCT is undefined via the real-view; the frontend deny-lists it.
        t = torch.full((128,), complex(self.rank + 1, 1), dtype=torch.complex64,
                       device=self.device)
        with self.assertRaises(Exception):
            c10d.all_reduce(t, op=c10d.ReduceOp.PRODUCT)

    def test_noncontiguous_rejects(self):
        # TCCL requires contiguous tensors; a non-contiguous view must reject.
        t = torch.full((64, 64), float(self.rank + 1), device=self.device).t()  # transpose -> non-contiguous
        self.assertFalse(t.is_contiguous())
        with self.assertRaises(Exception):
            c10d.all_reduce(t, op=c10d.ReduceOp.SUM)

    def test_stubbed_collectives_raise(self):
        # allreduce_coalesced is unimplemented and must raise a clean error at
        # dispatch, not crash. Every rank raises symmetrically.
        t = torch.ones(8, device=self.device)
        with self.assertRaises(Exception):
            c10d.all_reduce_coalesced([t], op=c10d.ReduceOp.SUM)

    # ---- ring topology: the two mesh-only capabilities must reject cleanly -----
    # Only when the group came up as a ring. Each reject is a synchronous
    # TORCH_CHECK at dispatch (before any RDMA), so every rank raises symmetrically.
    def test_ring_uneven_alltoall_rejects(self):
        if not _ring_topology():
            self.skipTest("ring-only: uneven alltoall is valid on a mesh")
        ws, r, dev = self.world_size, self.rank, self.device
        if ws <= 2:
            self.skipTest("ring uneven-alltoall reject needs world_size > 2")
        in_splits = [j + 1 for j in range(ws)]
        out_splits = [r + 1 for _ in range(ws)]
        inp = torch.full((sum(in_splits),), float(r + 1), dtype=torch.float32, device=dev)
        out = torch.empty(sum(out_splits), dtype=torch.float32, device=dev)
        with self.assertRaises(Exception):
            c10d.all_to_all_single(out, inp, out_splits, in_splits)

    def test_ring_nonneighbor_send_rejects(self):
        if not _ring_topology():
            self.skipTest("ring-only: non-neighbor send is valid on a mesh")
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 4:
            self.skipTest("a non-neighbor only exists at world_size >= 4 "
                          "(ws=3 ring is a fully-connected triangle)")
        dst = (r + 2) % ws  # two hops away -> never a ring neighbor for ws >= 4
        with self.assertRaises(Exception):
            c10d.send(torch.ones(64, device=dev), dst=dst)

    def test_ring_nonneighbor_recv_rejects(self):
        if not _ring_topology():
            self.skipTest("ring-only: non-neighbor recv is valid on a mesh")
        ws, r, dev = self.world_size, self.rank, self.device
        if ws < 4:
            self.skipTest("a non-neighbor only exists at world_size >= 4 "
                          "(ws=3 ring is a fully-connected triangle)")
        src = (r + 2) % ws
        with self.assertRaises(Exception):
            c10d.recv(torch.empty(64, device=dev), src=src)

    # gather / scatter are mesh-only; a ring rejects them synchronously at dispatch.
    def test_ring_gather_rejects(self):
        if not _ring_topology():
            self.skipTest("ring-only: gather is valid on a mesh")
        ws, r, dev, n = self.world_size, self.rank, self.device, 256
        inp = torch.full((n,), float(r + 1), device=dev)
        with self.assertRaises(Exception):
            if r == 0:
                c10d.gather(inp, gather_list=[torch.empty(n, device=dev) for _ in range(ws)], dst=0)
            else:
                c10d.gather(inp, gather_list=None, dst=0)

    def test_ring_scatter_rejects(self):
        if not _ring_topology():
            self.skipTest("ring-only: scatter is valid on a mesh")
        ws, r, dev, n = self.world_size, self.rank, self.device, 256
        out = torch.empty(n, device=dev)
        with self.assertRaises(Exception):
            if r == 0:
                c10d.scatter(out, scatter_list=[torch.full((n,), float(k + 1), device=dev)
                                                for k in range(ws)], src=0)
            else:
                c10d.scatter(out, scatter_list=None, src=0)


instantiate_parametrized_tests(ProcessGroupTcclCorrectnessTest)
instantiate_parametrized_tests(ProcessGroupTcclIntegrationTest)

if __name__ == "__main__":
    # The TCCL multi-node launcher appends --rank/--world-size/--master-addr/--port
    # for the legacy script convention; this suite reads them from the environment
    # instead (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT). Strip them so run_tests()'s
    # own argparse doesn't reject them.
    _strip = {"--rank", "--world-size", "--master-addr", "--port"}
    _argv, _it = [], iter(sys.argv[1:])
    for _a in _it:
        if _a in _strip:
            next(_it, None)  # also drop its value
            continue
        _argv.append(_a)
    sys.argv = [sys.argv[0]] + _argv
    run_tests()
