import torch
import torch.distributed as dist

from functorch import make_fx
from torch._C._distributed_c10d import _register_process_group
from torch._C._distributed_c10d import _create_work_from_future
from torch.futures import Future

import torch.distributed.traceable_collectives as tr_c


def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)

class LonelyRankProcessGroup(dist.ProcessGroup):
    def allreduce(self, tensors, opts):
        return ret_work(tensors)

    def __init__(self, rank, world):
        super(LonelyRankProcessGroup, self).__init__(rank, world)
        self._rank = rank
        self._world = world

    def size(self):
        return self._world

    def getBackendName(self):
        return "test"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"

def create_pg(prefix_store, rank, world_size, timeout):
    return LonelyRankProcessGroup(rank, world_size)
dist.Backend.register_backend('test', create_pg)
dist.init_process_group(backend="test", init_method="file:///tmp/init_pg", rank=0, world_size=1)

# FIXME this is a hack
pg_id = _register_process_group(dist.GroupMember.WORLD)

def fwd_fun(input):
    tmp =  tr_c.all_reduce(input + 1, reduceOp="sum", ranks=[0], tag="")
    tmp1 = input + 10
    return tmp + tmp1

print(make_fx(fwd_fun)(torch.rand(10)).graph)
