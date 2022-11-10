import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future, Store
from torch.futures import Future
from torch.utils._pytree import tree_flatten

"""
TODO:
Lots of missing collectives.
Collectives validation.
Make timeout robust by making collectives respect the test deadline.
Make tests robuts by making collectives interruptible.
We need some synchronization around cleanup to ensure that timedout ranks don't cause spurious failures.

"""


def flatten_list(lst):
    return tree_flatten(lst)[0]


def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)


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
    def __init__(self, world_size, collective):
        self._world_size = world_size
        self._collective = collective

        self._start_cond = threading.Condition()
        self._done_cond = threading.Condition()

        self._data = [None] * world_size
        self._count = 0
        self._done = False

    def join(self, rank, data):
        with self._start_cond:
            self._data[rank] = data
            self._count += 1

            # notify rank 0
            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()

            if rank == 0:
                while self._count < self._world_size:
                    self._start_cond.wait()

        with self._done_cond:
            # wait for rank 0 to finish
            if rank > 0:
                while not self._done:
                    self._done_cond.wait()
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
    _cur_coll = None

    @classmethod
    def _register(cls, pg):
        with cls._pg_lock:
            while len(cls._pg_list) <= pg._rank:
                cls._pg_list.append(None)
            cls._pg_list[pg._rank] = pg
            cls._count += 1
            if cls._count == pg._world:
                cls._ready = True

    @classmethod
    def _start_coll(cls, world_size, collective):
        with cls._coll_lock:
            if not cls._ready:
                raise Exception(
                    f"world not ready, only {cls._count} PG's registered but world has {world_size} ranks"
                )
            if cls._cur_coll is None:
                cls._cur_coll = Collective(world_size, collective)
            return cls._cur_coll

    @classmethod
    def _end_coll(cls, collective):
        # This is racily called by all ranks, so only one will work
        with cls._coll_lock:
            if cls._cur_coll == collective:
                cls._cur_coll = None

    def allgather(self, output_tensors, input_tensor, options):
        coll = ProcessLocalGroup._start_coll(self._world, AllGather())
        res = coll.join(self._rank, (output_tensors, input_tensor))
        ProcessLocalGroup._end_coll(coll)
        return res

    def broadcast(self, tensor_list, opts):
        coll = ProcessLocalGroup._start_coll(self._world, Broadcast(opts.rootRank))
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll)
        return res

    def __init__(self, rank, world):
        super(ProcessLocalGroup, self).__init__(rank, world)
        self._rank = rank
        self._world = world
        ProcessLocalGroup._register(self)

    def size(self):
        return self._world

    def getBackendName(self):
        return "local"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"


def _create_threaded_pg(prefix_store, rank, world_size, timeout):
    return ProcessLocalGroup(rank, world_size)


dist.Backend.register_backend("threaded", _create_threaded_pg)


@dataclass
class WorldData:
    default_pg: dist.ProcessGroup
    pg_map: Dict[dist.ProcessGroup, Tuple[str, Optional[Store]]]
    pg_names: Dict[dist.ProcessGroup, str]
    pg_group_ranks: Dict[dist.ProcessGroup, Dict[int, int]]
    group_count: int


class ThreadLocalWorld:
    _world = threading.local()

    def _get_world(self) -> WorldData:
        if not hasattr(ThreadLocalWorld._world, "world"):
            ThreadLocalWorld._world.world = WorldData(None, {}, {}, {}, 0)
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


def run_with_threaded_pg(world_size, timeout, callback):
    """
    Run ``callback`` with ``world_size`` threads using the in-proc process group
    """
    world = _install_threaded_pg()

    def world_is_valid():
        return world == dist.distributed_c10d._world

    global_store = dist.HashStore()
    exception_queue = queue.Queue()

    def worker(rank):
        if not world_is_valid():
            raise TimeoutError("Invalid world")
        dist.init_process_group(
            backend="threaded", rank=rank, world_size=world_size, store=global_store
        )
        try:
            callback()
        except BaseException as ex:
            exception_queue.put((rank, sys.exc_info()))
        finally:
            if world_is_valid():
                dist.destroy_process_group()

    try:
        threads = [
            threading.Thread(target=worker, args=(rank,)) for rank in range(world_size)
        ]
        for thread in threads:
            thread.start()

        deadline = time.time() + timeout
        for idx, thread in enumerate(threads):
            thread.join(max(0, deadline - time.time()))
            if thread.is_alive():
                exception_queue.put(
                    (
                        idx,
                        (
                            TimeoutError,
                            TimeoutError(
                                f"Rank failed to join in under {timeout} seconds"
                            ),
                            None,
                        ),
                    )
                )
        failed_ranks = []
        while not exception_queue.empty():
            failure = exception_queue.get()
            failed_ranks.append(failure)
        return failed_ranks
    finally:
        _uninstall_threaded_pg()
