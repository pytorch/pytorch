"""
vnccl: Virtual NCCL — a distributed backend that resolves collectives
locally using thread synchronization.

Designed for torchmux, where N distributed workers run as threads in a
single process sharing 1 GPU. Each collective blocks the calling thread
until all ranks have arrived, resolves the operation (allreduce = sum,
broadcast = copy from root, etc.), and unblocks all threads.

No GPUs are communicated between — all data movement is in-process
via tensor copy. The numerics are bitwise identical to real NCCL.

Usage:
    import torch.distributed.vnccl  # registers "vnccl" backend
    dist.init_process_group(backend="vnccl", rank=rank, world_size=N, store=store)

Typically used via torchmux rather than directly.

Note: this is distinct from multi_threaded_pg.ProcessLocalGroup in
torch/testing/_internal/ which serves a similar purpose for testing.
vnccl adds cooperative scheduling (_exec_lock), deterministic rank
ordering for trace reproducibility, per-worker RNG isolation, and
pairwise reduction to match NCCL's bitwise reduction order.
"""

import random as _random
import threading
import weakref
from functools import partial

import torch

__all__ = ["VNCCLProcessGroup"]

# Cooperative scheduling lock. When held, only one worker thread runs.
# Workers release it inside _do() before blocking on a collective
# barrier, allowing the next worker to run.
_exec_lock = threading.Lock()
_exec_lock_holder = threading.local()

# Deterministic rank ordering. After a collective resolves, rank 0
# runs first, then 1, 2, ..., producing a clean staircase in traces.
# NB: this is module-global, so it is only safe when a single process
# group is active at a time (the torchmux use case). Multiple concurrent
# groups would need per-group ordering state.
_next_rank = 0
_next_rank_cond = threading.Condition()


def _acquire_exec_lock_ordered(rank, world_size):
    with _next_rank_cond:
        _next_rank_cond.wait_for(lambda: _next_rank == rank)
    _exec_lock.acquire()
    _exec_lock_holder.held = True


def _release_exec_lock_ordered(rank, world_size):
    global _next_rank
    _exec_lock_holder.held = False
    with _next_rank_cond:
        _next_rank = (rank + 1) % world_size
        _next_rank_cond.notify_all()
    _exec_lock.release()


# Per-worker RNG state. Saved before yielding, restored after resuming.
# This gives each worker isolated RNG so manual_seed() in one worker
# doesn't corrupt another's random sequence.
_rng_states = {}


def _save_rng():
    state = {
        "cpu": torch.get_rng_state(),
        "random": _random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    _random.setstate(state["random"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


import torch.distributed as dist
from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    ScatterOptions,
)
from torch.distributed.distributed_c10d import _store_based_barrier
from torch.futures import Future
from torch.utils import _pytree as pytree


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _completed_work(result):
    fut = Future()
    fut.set_result(result)
    return _create_work_from_future(fut)


def _pairwise_reduce(tensors, op):
    """
    Reduce using pairwise in-place operations to match NCCL's reduction
    order. NCCL accumulates into tensor 0 by adding tensor 1, then
    tensor 2, etc. Using torch.stack + torch.sum would change the FP
    reduction order and break bitwise equivalence.
    """
    result = tensors[0].clone()
    for t in tensors[1:]:
        op(result, t)
    return result


_REDUCE_OPS = {
    ReduceOp.SUM: partial(_pairwise_reduce, op=lambda a, b: a.add_(b)),
    ReduceOp.AVG: partial(_pairwise_reduce, op=lambda a, b: a.add_(b)),
    ReduceOp.PRODUCT: partial(_pairwise_reduce, op=lambda a, b: a.mul_(b)),
    ReduceOp.MIN: partial(_pairwise_reduce, op=lambda a, b: torch.minimum(a, b, out=a)),
    ReduceOp.MAX: partial(_pairwise_reduce, op=lambda a, b: torch.maximum(a, b, out=a)),
    ReduceOp.BAND: partial(_pairwise_reduce, op=lambda a, b: a.bitwise_and_(b)),
    ReduceOp.BOR: partial(_pairwise_reduce, op=lambda a, b: a.bitwise_or_(b)),
    ReduceOp.BXOR: partial(_pairwise_reduce, op=lambda a, b: a.bitwise_xor_(b)),
}


# ------------------------------------------------------------------ #
# Collective resolution — each class has a work() method that takes
# the data contributed by all ranks and resolves the collective
# in-place.
# ------------------------------------------------------------------ #


class _AllReduce:
    def __init__(self, op):
        self.op = op.op

    @torch.no_grad()
    def work(self, data):
        for i in range(len(data[0])):
            dev = data[0][i].device
            tensors = [data[r][i].to(dev) for r in range(len(data))]
            res = _REDUCE_OPS[self.op](tensors)
            if self.op == ReduceOp.AVG:
                res.div_(len(data))
            for r in range(len(data)):
                data[r][i].detach().copy_(res.to(data[r][i].device))


class _Broadcast:
    def __init__(self, src):
        self.src = src

    @torch.no_grad()
    def work(self, data):
        src_tensors = pytree.tree_leaves(data[self.src])
        for i in range(len(data)):
            if i == self.src:
                continue
            dst_tensors = pytree.tree_leaves(data[i])
            for s, d in zip(src_tensors, dst_tensors):
                d.detach().copy_(s)


class _AllGather:
    @torch.no_grad()
    def work(self, data):
        for src_rank in range(len(data)):
            src_tensor = data[src_rank][1][0]
            for dest in data:
                dest[0][0][src_rank].detach().copy_(src_tensor)


class _ReduceScatter:
    def __init__(self, op):
        self.op = op if isinstance(op, int) else op.op

    @torch.no_grad()
    def work(self, data):
        ws = len(data)
        reduce_fn = _REDUCE_OPS[self.op]
        for i in range(ws):
            dst = data[i][0][0]
            chunks = [data[src][1][0][i].to(dst.device) for src in range(ws)]
            result = reduce_fn(chunks)
            if self.op == ReduceOp.AVG:
                result.div_(ws)
            dst.detach().copy_(result)


class _Scatter:
    def __init__(self, src):
        self.src = src

    @torch.no_grad()
    def work(self, data):
        src_tensors = data[self.src][1][0]
        for rank, each in enumerate(data):
            each[0][0].detach().copy_(src_tensors[rank])


class _Gather:
    def __init__(self, dst):
        self.dst = dst

    @torch.no_grad()
    def work(self, data):
        out_list = data[self.dst][0][0]
        for rank, each in enumerate(data):
            out_list[rank].detach().copy_(each[1][0])


class _AllToAll:
    @torch.no_grad()
    def work(self, data):
        ws = len(data)
        for dst in range(ws):
            out_list, _ = data[dst]
            for src in range(ws):
                _, in_list = data[src]
                out_list[src].detach().copy_(in_list[dst])


# ------------------------------------------------------------------ #
# Collective synchronization — blocks until all ranks have contributed
# their data, then the last rank to arrive resolves the operation and
# wakes everyone up.
# ------------------------------------------------------------------ #


_COLL_TIMEOUT_S = 300


class _CollSync:
    def __init__(self, world_size, op):
        self._world_size = world_size
        self._op = op
        self._cond = threading.Condition()
        self._data = [None] * world_size
        self._count = 0
        self._done = False
        self._error = None

    def join(self, rank, data):
        # Lock ordering: self._cond is always acquired before
        # _next_rank_cond. No code path acquires them in reverse.
        with self._cond:
            self._data[rank] = data
            self._count += 1
            if self._count < self._world_size:
                if not self._cond.wait_for(
                    lambda: self._done or self._error is not None,
                    timeout=_COLL_TIMEOUT_S,
                ):
                    raise RuntimeError(
                        f"rank {rank}: collective timed out after "
                        f"{_COLL_TIMEOUT_S}s waiting for all ranks "
                        f"({self._count}/{self._world_size} arrived)"
                    )
                if self._error is not None:
                    raise RuntimeError(
                        f"rank {rank}: collective failed on resolver"
                    ) from self._error
            else:
                try:
                    self._op.work(self._data)
                except Exception as e:
                    self._error = e
                    global _next_rank
                    with _next_rank_cond:
                        _next_rank = 0
                        _next_rank_cond.notify_all()
                    self._cond.notify_all()
                    raise
                self._done = True
                with _next_rank_cond:
                    _next_rank = 0
                    _next_rank_cond.notify_all()
                self._cond.notify_all()
        return _completed_work(data)


# ------------------------------------------------------------------ #
# VNCCLProcessGroup
# ------------------------------------------------------------------ #


class VNCCLProcessGroup(dist.ProcessGroup):
    """
    Process group that resolves collectives locally via thread sync.
    Multiple VNCCLProcessGroup instances (one per rank) coordinate
    through class-level shared state keyed by group name.
    """

    _lock = threading.Lock()
    _active = {}

    @classmethod
    def _enter(cls, op, pg, seq):
        with cls._lock:
            key = (pg.pg_name, seq)
            if key not in cls._active:
                cls._active[key] = _CollSync(pg.size(), op)
            return cls._active[key]

    @classmethod
    def _leave(cls, sync, pg, seq):
        with cls._lock:
            key = (pg.pg_name, seq)
            if cls._active.get(key) is sync:
                del cls._active[key]

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size
        self._coll_seq = 0

        world = dist.distributed_c10d._world
        # ThreadLocalWorld wraps _World for thread-based PGs; unwrap to
        # get the real _World so the weakref outlives each thread.
        if hasattr(world, "_get_world"):
            world = world._get_world()
        self._world = weakref.ref(world)

    def _do(self, op, data):
        seq = self._coll_seq
        self._coll_seq += 1
        held = getattr(_exec_lock_holder, "held", False)
        if held:
            _rng_states[self._rank] = _save_rng()
            _release_exec_lock_ordered(self._rank, self._world_size)
        try:
            sync = VNCCLProcessGroup._enter(op, self, seq)
            try:
                result = sync.join(self._rank, data)
            finally:
                VNCCLProcessGroup._leave(sync, self, seq)
        finally:
            if held:
                _acquire_exec_lock_ordered(self._rank, self._world_size)
                if self._rank in _rng_states:
                    _restore_rng(_rng_states[self._rank])
        return result

    # -- metadata --

    def size(self):
        return self._world_size

    @property
    def pg_name(self):
        world = self._world()
        if world is None:
            raise RuntimeError(
                "vnccl: distributed world has been destroyed; "
                "cannot use process group after teardown"
            )
        return world.pg_names[self]

    @property
    def group_name(self):
        return self.pg_name

    def getBackendName(self):
        return "vnccl"

    def __repr__(self):
        return f"VNCCL(rank={self._rank}, world_size={self._world_size})"

    # -- collectives --

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        return self._do(_AllReduce(opts.reduceOp), tensor_list)

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        return self._do(_AllReduce(opts.reduceOp), tensor_list)

    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        return self._do(_Broadcast(opts.rootRank), tensor_list)

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        return self._do(_AllGather(), (output_tensors, input_tensor))

    def _allgather_base(self, output, input, opts=AllgatherOptions()):
        chunks = list(torch.chunk(output, self._world_size))
        return self.allgather([chunks], [input], opts)

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=AllgatherOptions()):
        # Each tensor pair is a separate collective barrier rather than a
        # single batched operation. Correct but not performance-equivalent
        # to real NCCL's coalesced implementation.
        res = None
        for o, i in zip(outputs, inputs):
            res = self._allgather_base(o, i)
        return res

    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        return self._do(_Scatter(opts.rootRank), (output_tensors, input_tensors))

    def gather(self, output_tensors, input_tensors, opts=ScatterOptions()):
        return self._do(_Gather(opts.rootRank), (output_tensors, input_tensors))

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        return self._do(_ReduceScatter(opts.reduceOp), (output_tensor, scatter_list))

    def _reduce_scatter_base(self, output, input, opts=ReduceScatterOptions()):
        chunks = list(torch.chunk(input, self._world_size))
        return self.reduce_scatter([output], [chunks], opts)

    def reduce_scatter_tensor_coalesced(
        self, outputs, inputs, opts=ReduceScatterOptions()
    ):
        # Each tensor pair is a separate collective barrier rather than a
        # single batched operation. Correct but not performance-equivalent
        # to real NCCL's coalesced implementation.
        res = None
        for o, i in zip(outputs, inputs):
            res = self._reduce_scatter_base(o, i, opts)
        return res

    def alltoall(self, output_list, input_list, opts=AllToAllOptions()):
        return self._do(_AllToAll(), (output_list, input_list))

    def barrier(self, opts=BarrierOptions()):
        return self.allreduce(tensor_list=[torch.ones(1)])


def _create_vnccl(prefix_store, rank, world_size, timeout):
    pg = VNCCLProcessGroup(rank, world_size)
    _store_based_barrier(rank, prefix_store, "", world_size, timeout)
    return pg


if "vnccl" not in dist.Backend.backend_list:
    dist.Backend.register_backend("vnccl", _create_vnccl, devices=["cpu", "cuda"])
