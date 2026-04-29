"""
torchmux: Simulate N-GPU distributed training on M GPUs (M <= N).

Usage:
    python -m torch.distributed.torchmux --nproc-per-node 8 train.py [args...]
    python -m torch.distributed.torchmux --nproc-per-node 8 --ngpus 2 train.py [args...]

Launches N worker processes mapped round-robin onto M physical GPUs.
Workers on different GPUs run in parallel; workers sharing a GPU run
cooperatively, yielding at collective boundaries. When a worker yields
it checkpoints its entire CUDA context via the driver API (VRAM returned
to the driver) and restores it when rescheduled. Collectives are resolved
via the coordinator server — no NCCL is needed.

Produces two chrome://tracing traces (set TORCHMUX_TRACE_DIR):
a natural trace showing actual execution with per-GPU interleaving,
and a synthetic trace reconstructing what fully parallel execution
would look like.

Note [torchmux device remapping]:
Each worker process monkeypatches torch.device and torch.cuda.set_device
to transparently remap virtual CUDA device indices (e.g. cuda:5) onto
physical GPUs (e.g. cuda:1 when ngpus=2). Limitations:
  - Only affects Python-level calls to torch.device() made *after* the
    monkeypatch is installed. Modules that cache the original torch.device
    at import time (e.g. torch.nn.modules.module, torch.fx.graph) bypass
    the remapping.
  - isinstance(x, torch.device) works correctly via the _DeviceMeta
    metaclass, but `type(x) is torch.device` identity checks will fail
    (since actual device objects are _OrigDevice instances).
  - This is acceptable for standard training scripts that construct
    devices via torch.device(...) at call sites.
"""

__all__: list[str] = []

import argparse
import json
import logging
import os
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

log = logging.getLogger(__name__)


# ---- torch.device remapping ----

_OrigDevice = torch.device
_ngpus = 1


class _DeviceMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _OrigDevice)

    def __subclasscheck__(cls, subclass):
        if subclass is _OrigDevice:
            return True
        return type.__subclasscheck__(cls, subclass)


class _MuxDevice(metaclass=_DeviceMeta):
    def __new__(cls, *args, **kwargs):
        d = _OrigDevice(*args, **kwargs)
        if d.type == "cuda" and d.index is not None:
            return _OrigDevice("cuda", d.index % _ngpus)
        return d


# ---- Per-process trace recording ----

_MAX_TRACE_EVENTS = 1_000_000
_trace_events = []
_compute_start = None
_trace_path = None


def _us():
    return time.monotonic() * 1e6


def _trace(cat, name, start, dur):
    if len(_trace_events) < _MAX_TRACE_EVENTS:
        _trace_events.append((cat, name, start, dur))


def _begin_compute():
    global _compute_start
    _compute_start = _us()


def _end_compute():
    global _compute_start
    if _compute_start is not None:
        dur = _us() - _compute_start
        if dur > 0:
            _trace("compute", "compute", _compute_start, dur)
        _compute_start = None


def _flush_trace():
    if _trace_path and _trace_events:
        try:
            with open(_trace_path, "w") as f:
                json.dump(_trace_events, f)
        except Exception:
            pass


# ---- GPU state checkpoint/restore via CUDA driver ----

_client = None
_rank = None
_ws = None
_checkpointed = False
_coll_store = None
_coll_seq = 0


def _snapshot_gpu():
    global _checkpointed
    if _checkpointed or not torch.cuda.is_initialized():
        return
    import gc

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = _us()
    from torch.distributed._cuda_checkpoint import checkpoint_self

    checkpoint_self()
    _trace("mux", "snapshot", t0, _us() - t0)
    _checkpointed = True


def _restore_gpu():
    global _checkpointed
    if _checkpointed:
        t0 = _us()
        from torch.distributed._cuda_checkpoint import restore_self

        restore_self()
        _trace("mux", "restore", t0, _us() - t0)
        _checkpointed = False


# ---- Helpers ----


def _completed_work():
    from torch._C._distributed_c10d import _create_work_from_future
    from torch.futures import Future

    fut = Future()
    fut.set_result(None)
    return _create_work_from_future(fut)


# ---- Reduce operations ----

from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    ScatterOptions,
)


_REDUCE_FNS = {
    ReduceOp.SUM: torch.Tensor.add_,
    ReduceOp.AVG: torch.Tensor.add_,
    ReduceOp.PREMUL_SUM: torch.Tensor.add_,
    ReduceOp.PRODUCT: torch.Tensor.mul_,
    ReduceOp.MIN: lambda a, b: torch.minimum(a, b, out=a),
    ReduceOp.MAX: lambda a, b: torch.maximum(a, b, out=a),
    ReduceOp.BAND: torch.Tensor.bitwise_and_,
    ReduceOp.BOR: torch.Tensor.bitwise_or_,
    ReduceOp.BXOR: torch.Tensor.bitwise_xor_,
}


def _get_reduce_fn(op):
    if hasattr(op, "op"):
        op = op.op
    return _REDUCE_FNS[op]


def _is_avg(op):
    if hasattr(op, "op"):
        op = op.op
    return op == ReduceOp.AVG


# ---- Coordinator-backed ProcessGroup ----


_pending_group_ranks = None


class _MuxPG(dist.ProcessGroup):
    """Resolves collectives via the coordinator's prepare() primitive.

    Each collective: serialize tensors → send via prepare() → if all
    peers have deposited, get data immediately (fast path); otherwise
    checkpoint GPU, release to coordinator, wait, restore, get data.
    """

    _next_pg_id = 0

    def __init__(self, rank, world_size, group_ranks=None):
        super().__init__(rank, world_size)
        self._rank = rank
        self._ws = world_size
        self._group_ranks = group_ranks or list(range(world_size))
        self._pg_tag = "_".join(str(r) for r in sorted(self._group_ranks))
        self._seq = 0

    @property
    def _my_global(self):
        return self._group_ranks[self._rank]

    @property
    def _others(self):
        me = self._my_global
        return tuple(r for r in self._group_ranks if r != me)

    def _do_prepare(self, send, recv):
        import json

        from torch.distributed._coord_client import (
            _deserialize_tensor,
            _serialize_tensor,
        )

        seq = self._seq
        self._seq += 1
        tag = self._pg_tag
        me = self._my_global

        for dsts, tensor in send.items():
            if tensor is not None:
                hdr, body = _serialize_tensor(tensor)
                meta = json.dumps(hdr).encode()
            else:
                meta = b"null"
                body = b""
            for dst in dsts:
                key = f"g{tag}s{seq}_{me}_{dst}"
                _coll_store.set(key + "h", meta)
                _coll_store.set(key + "d", body)

        if _ws > _ngpus:
            _snapshot_gpu()
            _client.release_baton()

        result = {}
        for src in recv:
            key = f"g{tag}s{seq}_{src}_{me}"
            hdr_bytes = _coll_store.get(key + "h")
            if hdr_bytes == b"null":
                result[src] = None
            else:
                data = _coll_store.get(key + "d")
                hdr = json.loads(hdr_bytes.decode())
                result[src] = _deserialize_tensor(hdr, data)

        if _ws > _ngpus:
            _client.acquire_baton()
            _restore_gpu()

        return result

    def _do_collective(self, name, fn):
        _end_compute()
        t0 = _us()
        with torch.profiler.record_function(f"torchmux::collective::{name}"):
            result = fn()
        _trace("collective", name, t0, _us() - t0)
        _flush_trace()
        _begin_compute()
        return result

    # ---- collectives ----

    @torch.no_grad()
    def allreduce(self, tensor_list, opts=None):
        if opts is None:
            opts = AllreduceOptions()

        def _run():
            op = opts.reduceOp if hasattr(opts, "reduceOp") else ReduceOp.SUM
            others = self._others
            if not others:
                return _completed_work()
            reduce_fn = _get_reduce_fn(op)
            for t in tensor_list:
                result = self._do_prepare({others: t}, others)
                for src in others:
                    reduce_fn(t, result[src].to(t.device))
                if _is_avg(op):
                    t.div_(self._ws)
            return _completed_work()

        return self._do_collective("allreduce", _run)

    @torch.no_grad()
    def allreduce_coalesced(self, tensor_list, opts=None):
        return self.allreduce(tensor_list, opts)

    @torch.no_grad()
    def broadcast(self, tensor_list, opts=None):
        if opts is None:
            opts = BroadcastOptions()

        def _run():
            root = opts.rootRank
            root_global = self._group_ranks[root]
            me = self._my_global
            others = self._others
            if not others:
                return _completed_work()
            if me == root_global:
                for t in tensor_list:
                    self._do_prepare({others: t}, ())
            else:
                for t in tensor_list:
                    result = self._do_prepare({}, (root_global,))
                    t.copy_(result[root_global].to(t.device))
            return _completed_work()

        return self._do_collective("broadcast", _run)

    @torch.no_grad()
    def allgather(self, output_tensors, input_tensor, opts=None):
        if opts is None:
            opts = AllgatherOptions()

        def _run():
            others = self._others
            me = self._my_global
            for i, t in enumerate(input_tensor):
                if not others:
                    output_tensors[i][0].copy_(t)
                    continue
                result = self._do_prepare({others: t}, others)
                for local_r in range(self._ws):
                    global_r = self._group_ranks[local_r]
                    if global_r == me:
                        output_tensors[i][local_r].copy_(t)
                    else:
                        output_tensors[i][local_r].copy_(
                            result[global_r].to(output_tensors[i][local_r].device)
                        )
            return _completed_work()

        return self._do_collective("allgather", _run)

    @torch.no_grad()
    def _allgather_base(self, output, input, opts=None):
        if opts is None:
            opts = AllgatherOptions()

        def _run():
            others = self._others
            me = self._my_global
            if not others:
                output.copy_(input)
                return _completed_work()
            result = self._do_prepare({others: input}, others)
            chunk_size = input.size(0)
            for local_r in range(self._ws):
                global_r = self._group_ranks[local_r]
                dst = output[local_r * chunk_size : (local_r + 1) * chunk_size]
                if global_r == me:
                    dst.copy_(input)
                else:
                    dst.copy_(result[global_r].to(output.device))
            return _completed_work()

        return self._do_collective("allgather", _run)

    @torch.no_grad()
    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):
        if opts is None:
            opts = AllgatherOptions()
        for o, i in zip(outputs, inputs):
            self._allgather_base(o, i, opts)
        return _completed_work()

    @torch.no_grad()
    def _reduce_scatter_base(self, output, input, opts=None):
        if opts is None:
            opts = ReduceScatterOptions()

        def _run():
            op = opts.reduceOp if hasattr(opts, "reduceOp") else ReduceOp.SUM
            others = self._others
            me = self._my_global
            chunk_size = input.size(0) // self._ws
            my_chunk = input[
                self._rank * chunk_size : (self._rank + 1) * chunk_size
            ]
            if not others:
                output.copy_(my_chunk)
                return _completed_work()
            send = {}
            for local_r in range(self._ws):
                global_r = self._group_ranks[local_r]
                if global_r != me:
                    chunk = input[
                        local_r * chunk_size : (local_r + 1) * chunk_size
                    ]
                    send[(global_r,)] = chunk
            result = self._do_prepare(send, others)
            reduce_fn = _get_reduce_fn(op)
            acc = my_chunk.detach().cpu().clone()
            for src in others:
                reduce_fn(acc, result[src])
            if _is_avg(op):
                acc.div_(self._ws)
            output.copy_(acc.to(output.device))
            return _completed_work()

        return self._do_collective("reduce_scatter", _run)

    @torch.no_grad()
    def reduce_scatter(self, output_tensor, scatter_list, opts=None):
        if opts is None:
            opts = ReduceScatterOptions()

        def _run():
            op = opts.reduceOp if hasattr(opts, "reduceOp") else ReduceOp.SUM
            reduce_fn = _get_reduce_fn(op)
            me = self._my_global
            others = self._others
            for i, chunks in enumerate(scatter_list):
                if not others:
                    output_tensor[i].copy_(chunks[self._rank])
                    continue
                send = {}
                for local_r in range(self._ws):
                    global_r = self._group_ranks[local_r]
                    if global_r != me:
                        send[(global_r,)] = chunks[local_r]
                result = self._do_prepare(send, others)
                acc = chunks[self._rank].detach().cpu().clone()
                for src in others:
                    reduce_fn(acc, result[src])
                if _is_avg(op):
                    acc.div_(self._ws)
                output_tensor[i].copy_(acc.to(output_tensor[i].device))
            return _completed_work()

        return self._do_collective("reduce_scatter", _run)

    @torch.no_grad()
    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):
        if opts is None:
            opts = ReduceScatterOptions()
        for o, i in zip(outputs, inputs):
            self._reduce_scatter_base(o, i, opts)
        return _completed_work()

    @torch.no_grad()
    def scatter(self, output, input, opts=None):
        raise NotImplementedError("_MuxPG.scatter")

    @torch.no_grad()
    def gather(self, output, input, opts=None):
        raise NotImplementedError("_MuxPG.gather")

    @torch.no_grad()
    def alltoall(self, output, input, opts=None):
        raise NotImplementedError("_MuxPG.alltoall")

    @torch.no_grad()
    def barrier(self, opts=None):
        if opts is None:
            opts = BarrierOptions()

        def _run():
            others = self._others
            if not others:
                return _completed_work()
            self._do_prepare({others: None}, others)
            return _completed_work()

        return self._do_collective("barrier", _run)

    # ---- metadata ----

    def size(self):
        return self._ws

    def getBackendName(self):
        return "mux_coord"

    @property
    def pg_name(self):
        return dist.distributed_c10d._world.pg_names.get(
            self, f"mux_pg_{id(self)}"
        )

    @property
    def group_name(self):
        return self.pg_name

    def __repr__(self):
        return f"MuxCoord(rank={self._rank}, size={self._ws})"


# ---- Backend factory ----


def _create_mux_pg(store, rank, world_size, timeout):
    from torch.distributed.distributed_c10d import _store_based_barrier

    group_ranks = _pending_group_ranks
    if group_ranks is None:
        group_ranks = list(range(world_size))

    store.set(f"torchmux_grank_{rank}", str(_rank).encode())

    # Callers (_mux_init, _mux_new_group) already handle snapshot/restore
    # and baton release/acquire. Just do the barrier here.
    _store_based_barrier(rank, store, "", world_size, timeout)

    if len(group_ranks) != world_size:
        group_ranks = []
        for r in range(world_size):
            group_ranks.append(int(store.get(f"torchmux_grank_{r}")))

    pg = _MuxPG(rank, world_size, group_ranks)
    return pg


# ---- Worker process ----


def _worker(
    rank,
    world_size,
    ngpus,
    master_port,
    coord_addr,
    trace_dir_path,
    script,
    script_args,
    run_as_module,
    coll_store_port=0,
):
    global _ngpus, _rank, _ws, _client, _coll_store

    _ngpus = ngpus
    _rank = rank
    _ws = world_size

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "GROUP_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "TORCHMUX_CUDA_VISIBLE_DEVICES",
                ",".join(str(i) for i in range(ngpus)),
            ),
        }
    )

    if coll_store_port:
        _coll_store = dist.TCPStore(
            "localhost",
            coll_store_port,
            world_size,
            is_master=(rank == 0),
        )

    torch.device = _MuxDevice
    _orig_cuda_set_device = torch.cuda.set_device
    torch.cuda.set_device = lambda *a, **kw: _orig_cuda_set_device(rank % ngpus)

    from torch.distributed._coord_client import CoordClient

    _client = CoordClient(addr=coord_addr)
    _client.register(rank, gpu_id=rank % ngpus)
    _client.wait_for_turn()

    dist.Backend.register_backend(
        "mux_coord", _create_mux_pg, devices=["cpu", "cuda"]
    )

    _orig_init = dist.init_process_group
    _orig_destroy = dist.destroy_process_group
    _orig_new_group = dist.new_group

    def _mux_init(backend=None, **kwargs):
        global _pending_group_ranks
        _end_compute()
        t0 = _us()
        _pending_group_ranks = list(range(world_size))
        _snapshot_gpu()
        _client.release_baton()
        try:
            _orig_init(backend="mux_coord", **kwargs)
        finally:
            _client.acquire_baton()
            _restore_gpu()
            _pending_group_ranks = None
        _trace("collective", "init", t0, _us() - t0)
        _begin_compute()

    def _mux_destroy():
        _end_compute()

        try:
            with open(trace_path, "w") as f:
                json.dump(_trace_events, f)
        except Exception:
            pass

        t0 = _us()
        _snapshot_gpu()
        _client.release_baton()
        try:
            _orig_destroy()
        finally:
            _client.acquire_baton()
            _restore_gpu()
            _trace("collective", "destroy", t0, _us() - t0)
            _begin_compute()

    def _mux_new_group(ranks=None, *args, **kwargs):
        global _pending_group_ranks
        _end_compute()
        t0 = _us()
        if ranks is None:
            ranks = list(range(world_size))
        _pending_group_ranks = list(ranks)
        _snapshot_gpu()
        _client.release_baton()
        try:
            result = _orig_new_group(ranks, *args, **kwargs)
        finally:
            _client.acquire_baton()
            _restore_gpu()
            _pending_group_ranks = None
            _trace("collective", "new_group", t0, _us() - t0)
            _begin_compute()
        return result

    dist.init_process_group = _mux_init
    dist.destroy_process_group = _mux_destroy
    dist.new_group = _mux_new_group

    trace_dir = os.environ.get("TORCHMUX_TRACE_DIR", "")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    # The CUDA checkpoint API cannot checkpoint/restore state created by
    # flash attention or cuDNN SDPA kernels. Force the math-only backend
    # so all SDPA operations use checkpoint-safe CUDA kernels.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    global _trace_path
    trace_path = os.path.join(trace_dir_path, f"trace_rank{rank}.json")
    _trace_path = trace_path

    _begin_compute()
    try:
        filtered_args = [a for a in script_args if a != "--"]
        sys.argv = [script] + filtered_args
        if run_as_module:
            runpy.run_module(script, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    finally:
        _end_compute()

        try:
            with open(trace_path, "w") as f:
                json.dump(_trace_events, f)
        except Exception:
            pass

        os._exit(0)


# ---- Entry point ----


def _print_timing_summary(events_by_rank):
    header = (
        f"{'Worker':>8} {'Wall(s)':>9} {'Compute%':>9} "
        f"{'Snap%':>9} {'Restore%':>9} {'Overhead%':>9}"
    )
    log.info("\ntorchmux: timing overview\n%s", header)

    sum_wall = 0
    sum_compute = 0
    sum_snap = 0
    sum_restore = 0

    for r in sorted(events_by_rank):
        evts = events_by_rank[r]
        if not evts:
            continue
        wall = max(s + d for _, _, s, d in evts) - min(s for _, _, s, d in evts)
        if wall <= 0:
            continue
        compute = sum(d for cat, _, _, d in evts if cat == "compute")
        snap = sum(d for _, name, _, d in evts if name == "snapshot")
        restore = sum(d for _, name, _, d in evts if name == "restore")
        overhead = snap + restore

        sum_wall += wall
        sum_compute += compute
        sum_snap += snap
        sum_restore += restore

        log.info(
            "%s %s %s %s %s %s",
            f"{'R' + str(r):>8}",
            f"{wall / 1e6:>8.1f}s",
            f"{compute / wall * 100:>8.1f}%",
            f"{snap / wall * 100:>8.1f}%",
            f"{restore / wall * 100:>8.1f}%",
            f"{overhead / wall * 100:>8.1f}%",
        )

    if len(events_by_rank) > 1 and sum_wall > 0:
        avg_wall = sum_wall / len(events_by_rank)
        log.info(
            "%s %s %s %s %s %s",
            f"{'Avg':>8}",
            f"{avg_wall / 1e6:>8.1f}s",
            f"{sum_compute / sum_wall * 100:>8.1f}%",
            f"{sum_snap / sum_wall * 100:>8.1f}%",
            f"{sum_restore / sum_wall * 100:>8.1f}%",
            f"{(sum_snap + sum_restore) / sum_wall * 100:>8.1f}%",
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="torchmux",
        description="Simulate N-GPU distributed training on M GPUs",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        required=True,
        dest="nproc_per_node",
        help="Number of simulated workers",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of physical GPUs (default 1)",
    )
    parser.add_argument(
        "-m",
        "--module",
        action="store_true",
        dest="module",
        help="Run script as a Python module",
    )
    parser.add_argument("training_script", help="Training script or module name")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = args.nproc_per_node
    ngpus = args.ngpus
    if nproc < 1:
        parser.error("--nproc-per-node must be >= 1")
    if ngpus < 1:
        parser.error("--ngpus must be >= 1")
    if ngpus > nproc:
        parser.error(
            f"--ngpus ({ngpus}) must be <= --nproc-per-node ({nproc})"
        )

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    with socket.socket() as s:
        s.bind(("", 0))
        coll_store_port = s.getsockname()[1]

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if shm is None:
        log.warning(
            "torchmux: /dev/shm not available, using disk-backed tmpdir for "
            "trace data."
        )
    trace_tmpdir = tempfile.mkdtemp(prefix="torchmux_traces_", dir=shm)

    # Propagate parent CUDA_VISIBLE_DEVICES to workers via env var so
    # workers use the correct physical GPUs.
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd is not None:
        os.environ["TORCHMUX_CUDA_VISIBLE_DEVICES"] = parent_cvd

    # Start coordinator subprocess.
    initial_holders = {g: g for g in range(ngpus)}
    coord_cmd = [
        sys.executable,
        "-m",
        "torch.distributed._coordinator",
        "--tcp-port",
        "0",
        "--initial-holders",
        json.dumps(initial_holders),
    ]
    coord_proc = subprocess.Popen(
        coord_cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )

    try:
        addr_line = coord_proc.stdout.readline().decode().strip()
        if not addr_line.startswith("ADDR "):
            coord_proc.kill()
            raise RuntimeError(
                f"coordinator failed to start: {addr_line!r}"
            )
        coord_addr = addr_line.split(" ", 1)[1]

        gpu_desc = "GPU 0" if ngpus == 1 else f"{ngpus} GPUs"
        log.info(
            "torchmux: %d workers on %s, script=%s, coordinator=%s",
            nproc,
            gpu_desc,
            args.training_script,
            coord_addr,
        )

        mp.spawn(
            _worker,
            args=(
                nproc,
                ngpus,
                port,
                coord_addr,
                trace_tmpdir,
                args.training_script,
                args.training_script_args,
                args.module,
                coll_store_port,
            ),
            nprocs=nproc,
            join=True,
        )
    finally:
        coord_proc.terminate()
        try:
            coord_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            coord_proc.kill()
            coord_proc.wait()

        from torch.distributed import torchmux_trace

        trace_dir = os.environ.get("TORCHMUX_TRACE_DIR", "")

        events_by_rank = {}
        for r in range(nproc):
            p = os.path.join(trace_tmpdir, f"trace_rank{r}.json")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    with open(p) as f:
                        events_by_rank[r] = [tuple(e) for e in json.load(f)]
                except (json.JSONDecodeError, ValueError):
                    pass

        if events_by_rank and trace_dir:
            natural_path = os.path.join(trace_dir, "torchmux_natural.json")
            synthetic_path = os.path.join(trace_dir, "torchmux_synthetic.json")

            torchmux_trace.export_natural(events_by_rank, natural_path, nproc)
            torchmux_trace.export_synthetic(
                events_by_rank, synthetic_path, nproc
            )

            log.info(
                "torchmux: traces written to %s and %s",
                natural_path,
                synthetic_path,
            )

        if events_by_rank:
            _print_timing_summary(events_by_rank)

        shutil.rmtree(trace_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
