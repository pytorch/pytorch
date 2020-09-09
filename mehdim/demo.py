import logging
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    group = dist.new_group([0, 1])
    async_op = True
    with torch.autograd.profiler.profile() as prof:
        tensor = torch.ones(128)
        work = dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group, async_op=async_op)
        if async_op:
            work.wait()
            work._get_profiling_future().wait()
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
