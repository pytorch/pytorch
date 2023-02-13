import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllgatherOptions,
    AllreduceOptions,
    BroadcastOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    ReduceOp,
)
from torch.futures import Future
from torch.utils._pytree import tree_flatten

"""
TODO:
Lots of missing collectives.
Collectives validation.
Make timeout robust by making collectives respect the test deadline.
Make tests robust by making collectives interruptible.
We need some synchronization around cleanup to ensure that timedout ranks don't cause spurious failures.

"""


def flatten_list(lst):
    return tree_flatten(lst)[0]


def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)


class AllReduce:
    def __init__(self, op):
        if op != ReduceOp.SUM:
            raise NotImplementedError(
                "AllReduce only supports SUM on threaded pg for now."
            )
        self.op = op

    def work(self, data):
        # data: List[List[Tensor]]
        res = data[0][0]
        for src_rank in range(1, len(data)):
            in_tensor_list = data[src_rank]
            res.add_(in_tensor_list[0])  # Hardcoded
        with torch.no_grad():
            for src_rank in range(len(data)):
                data[src_rank][0].copy_(res)


class AllGather:
    def work(self, data):
        for src_rank in range(len(data)):
            in_tensor_list = data[src_rank][1]
            # Can't handle all_gather with multiple tensors
            assert len(in_tensor_list) == 1
            src_tensor = in_tensor_list[0]

            for dest in data:
                dest_tensor = dest[0][0][src_rank]
                with torch.no_grad():
                    dest_tensor.copy_(src_tensor)

class Scatter:
    def __init__(self, src):
        self.src = src

    def work(self, data):
        src_in_tensor_list = data[self.src][1]
        # Can't handle scatter with multiple input tensor list
        assert len(src_in_tensor_list) == 1
        src_in_tensors = src_in_tensor_list[0]

        for rank, each_rank_data in enumerate(data):
            out_tensor_list = each_rank_data[0]
            # Can't handle scatter with multiple output tensor
            assert len(out_tensor_list) == 1
            dest_tensor = out_tensor_list[0]
            with torch.no_grad():
                dest_tensor.copy_(src_in_tensors[rank])

class ReduceScatter:
    def __init__(self, op):
        if op != dist.ReduceOp.SUM:
            raise NotImplementedError("ReduceScatter only supports SUM on threaded pg for now.")
        self.op = op

    def work(self, data):
        start_reduction = [False for _ in range(len(data))]
        for each_rank_data in data:
            # Can't handle reduce_scatter with multiple scatter list
            assert len(each_rank_data[1]) == 1
            to_scatter = each_rank_data[1][0]
            for i in range(len(to_scatter)):
                dest_tensor_on_rank_i = data[i][0]
                # Can't handle reduce_scatter with multiple output tensor
                assert len(dest_tensor_on_rank_i) == 1
                if not start_reduction[i]:
                    with torch.no_grad():
                        dest_tensor_on_rank_i[0].copy_(to_scatter[i])
                    start_reduction[i] = True
                else:
                    with torch.no_grad():
                        dest_tensor_on_rank_i[0].add_(to_scatter[i])

class Broadcast:
    def __init__(self, src):
        self.src = src

    def work(self, data):
        in_tensor_list = flatten_list(data[self.src])
        for i in range(len(data)):
            out_tensor_list = flatten_list(data[i])
            for j in range(len(in_tensor_list)):
                with torch.no_grad():
                    out_tensor_list[j].copy_(in_tensor_list[j])


class Collective:
    def __init__(self, world_size, collective, pg):
        self._world_size = world_size
        self._collective = collective

        self._start_cond = threading.Condition()
        self._done_cond = threading.Condition()

        self._data = [None] * world_size
        self._count = 0
        self._done = False

        self._pg = pg

    def join(self, rank, data):
        with self._start_cond:
            self._data[rank] = data
            self._count += 1

            # notify rank 0
            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()

            if rank == 0:
                self._start_cond.wait_for(
                    lambda: self._count == self._world_size or self._pg._terminate.is_set()
                )
                # SystemExit is not a subclass of Exception but BaseException
                # and can be distinguished from normal exception raised from program errors
                # so that we can hide it from the exception queue
                if self._pg._terminate.is_set():
                    sys.exit("Test termination event occurs.")

        with self._done_cond:
            # wait for rank 0 to finish
            if rank > 0:
                self._done_cond.wait_for(lambda: self._done or self._pg._terminate.is_set())
                if self._pg._terminate.is_set():
                    sys.exit("Test termination event occurs.")
            else:
                # copy data around
                self._collective.work(self._data)
                self._done = True
                self._done_cond.notify_all()
        return ret_work(data)


class ProcessLocalGroup(dist.ProcessGroup):
    _pg_lock = threading.Lock()
    _pg_list = []
    _count = 0
    _ready = False

    _coll_lock = threading.Lock()
    _cur_coll_on_pgs = {}

    _terminate = threading.Event()

    @classmethod
    def _register(cls, pg):
        with cls._pg_lock:
            while len(cls._pg_list) <= pg._rank:
                cls._pg_list.append(None)
            cls._pg_list[pg._rank] = pg
            cls._count += 1
            if cls._count == pg._world_size:
                cls._ready = True

    @classmethod
    def _start_coll(cls, collective, pg):
        with cls._coll_lock:
            if not cls._ready:
                raise Exception(
                    f"world not ready, only {cls._count} PG's registered but world has {pg.size()} ranks"
                )
            # pg_name is unique, we use that to record the mapping between pg and collective
            if pg.pg_name not in cls._cur_coll_on_pgs:
                cls._cur_coll_on_pgs[pg.pg_name] = Collective(pg.size(), collective, cls)
            return cls._cur_coll_on_pgs[pg.pg_name]

    @classmethod
    def _end_coll(cls, collective, pg):
        # This is racily called by all ranks, so only one will work
        with cls._coll_lock:
            if pg.pg_name in cls._cur_coll_on_pgs and cls._cur_coll_on_pgs[pg.pg_name] == collective:
                cls._cur_coll_on_pgs.pop(pg.pg_name)

    @classmethod
    def exception_handle(cls, exc):
        cls._terminate.set()
        for coll in cls._cur_coll_on_pgs.values():
            with coll._start_cond:
                coll._start_cond.notify()
            with coll._done_cond:
                coll._done_cond.notify_all()

    @classmethod
    def reset(cls):
        with cls._coll_lock:
            cls._cur_coll_on_pgs = {}
            cls._terminate.clear()

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        coll = ProcessLocalGroup._start_coll(AllGather(), self)
        res = coll.join(self._rank, (output_tensors, input_tensor))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        coll = ProcessLocalGroup._start_coll(Broadcast(opts.rootRank), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        coll = ProcessLocalGroup._start_coll(Scatter(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        coll = ProcessLocalGroup._start_coll(ReduceScatter(opts.reduceOp), self)
        res = coll.join(self._rank, (output_tensor, scatter_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size
        ProcessLocalGroup._register(self)

    def size(self):
        return self._world_size

    @property
    def pg_name(self):
        """
        return the global registered name of the current pg in the world
        """
        return dist.distributed_c10d._world.pg_names[self]

    def getBackendName(self):
        return "threaded"

    def __repr__(self):
        return f"ThreadedPG world_size:{self._world_size} rank:{self._rank}"


def _create_threaded_pg(prefix_store, rank, world_size, timeout):
    return ProcessLocalGroup(rank, world_size)


dist.Backend.register_backend("threaded", _create_threaded_pg)


@dataclass
class WorldData:
    default_pg: dist.ProcessGroup
    pg_map: Dict[dist.ProcessGroup, Tuple[str, Optional[Store]]]
    pg_names: Dict[dist.ProcessGroup, str]
    pg_group_ranks: Dict[dist.ProcessGroup, Dict[int, int]]
    pg_backend_config: Dict[dist.ProcessGroup, str]
    group_count: int


class ThreadLocalWorld:
    _world = threading.local()

    def _get_world(self) -> WorldData:
        if not hasattr(ThreadLocalWorld._world, "world"):
            ThreadLocalWorld._world.world = WorldData(None, {}, {}, {}, {}, 0)
        return ThreadLocalWorld._world.world

    @property
    def default_pg(self):
        return self._get_world().default_pg

    @default_pg.setter
    def default_pg(self, value):
        self._get_world().default_pg = value

    @property
    def pg_map(self):
        return self._get_world().pg_map

    @property
    def pg_names(self):
        return self._get_world().pg_names

    @property
    def pg_group_ranks(self):
        return self._get_world().pg_group_ranks

    @property
    def pg_backend_config(self):
        return self._get_world().pg_backend_config

    @property
    def group_count(self) -> int:
        return self._get_world().group_count

    @group_count.setter
    def group_count(self, value):
        self._get_world().group_count = value


_old_pg_world = None


def _install_threaded_pg():
    global _old_pg_world
    _old_pg_world = dist.distributed_c10d._world
    dist.distributed_c10d._world = ThreadLocalWorld()
    return dist.distributed_c10d._world


def _uninstall_threaded_pg():
    dist.distributed_c10d._world = _old_pg_world
