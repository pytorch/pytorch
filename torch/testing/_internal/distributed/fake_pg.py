import os

import torch.distributed as dist

from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllgatherOptions,
    AllreduceOptions,
    BarrierOptions,
    ReduceScatterOptions
)
from torch.futures import Future


def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)


class FakeProcessGroup(dist.ProcessGroup):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convinient tool when playing
    with distributed but don't care about the actual data.
    """
    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        return ret_work(tensor_list)

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        return ret_work(output_tensors)

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        return ret_work(output_tensor)

    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        # assume each rank have the same input tensor so we just copy to the results
        # since it's not a real allgather, we simply make this copying logic to let
        # some simple validation works (i.e. calling allgather to see if each rank have
        # the same tensor or not)
        # NOTE: in general it's not good form to try to make FakePG work with 'real data',
        # but the reasoning here is that we want FakePG to work with DeviceMesh's init
        # code that have the data validation, which makes it worth the tradeoff.
        # In general user should use MTPG or normal PG for cases where they may care about
        # real data from collectives
        chunks = output_tensor.chunk(self._world_size)
        for chunk in chunks:
            chunk.copy_(input_tensor)
        return ret_work(output_tensor)

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=ReduceScatterOptions()):
        return ret_work(output_tensor)

    def barrier(self, opts=BarrierOptions()):
        # it should be no-op for fake pg
        pass

    def getBackendName(self):
        return "fake"

    def __repr__(self):
        return f"FakePG world_size:{self._world_size} rank:{self._rank}"


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """
    pass

def _create_fake_pg(prefix_store, rank, world_size, timeout):
    # disable barrier after init for fake pg, as it does not
    # need to sync with other processes/threads
    os.environ["TORCH_DIST_INIT_BARRIER"] = "0"
    return FakeProcessGroup(rank, world_size)

dist.Backend.register_backend("fake", _create_fake_pg, devices=['cpu', 'cuda'])
