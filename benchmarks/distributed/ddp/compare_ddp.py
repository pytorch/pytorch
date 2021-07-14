"""
A simple tool to compare the performance of different impls of
DistributedDataParallel, four flavors:

1. DistributedDataParallel, which has a python wrapper and C++ core to do
   gradient distribution and reduction. It's current production version.

2. LegacyDistributedDataParallel: a pure python implementation which relies
   on synchronous all_reduce.

3. PythonDDP, a pure python implementation which tries to replicate all the
   behaviors of 1. The goal is to have on-par performance with 1 so that we
   can move DistributedDataParallel to python-only implementation to simplify
   the stack.

Example::
    >>> modify configs in main func
    >>> python compare_ddp.py
    >>> Sample out: compare_ddp_sample.md
"""

import legacy_distributed_data_parallel as legacy_ddp
import numpy as np
import os
import python_ddp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from enum import Enum
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPOption(Enum):
    DDP_CPP_CORE = 1
    LEGACY_DISTRIBUTED_DATA_PARALLEL = 2
    PYTHON_DDP_SYNC = 3
    PYTHON_DDP_ASYNC = 4

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def create_ddp_model(module, rank, pg, ddp_option, buffer_size):
    """Helper to create DDPModel. """
    if ddp_option == DDPOption.DDP_CPP_CORE:
        ddp_model = DDP(module, device_ids=[rank], process_group=pg)
        ddp_model._set_static_graph()
        return ddp_model
    elif ddp_option == DDPOption.LEGACY_DISTRIBUTED_DATA_PARALLEL:
        return legacy_ddp.LegacyDistributedDataParallel(module, pg, buffer_size)
    elif ddp_option == DDPOption.PYTHON_DDP_SYNC:
        return python_ddp.PythonDDP(module, pg, False, buffer_size)
    elif ddp_option == DDPOption.PYTHON_DDP_ASYNC:
        return python_ddp.PythonDDP(module, pg, True, buffer_size)
    else:
        raise NotImplementedError

def run_ddp(rank, world_size, epochs, ddp_option, buffer_size):
    print(f'Invoked run_ddp rank {rank}')

    # Setup
    print("setting up ... ")
    setup(rank, world_size)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    device = torch.device('cuda:%d' % rank)
    print('done')

    # Create ResNet50 module and wrap in DDP module.
    pg = dist.distributed_c10d._get_default_group()
    model = models.resnet50().to(device)
    ddp_model = create_ddp_model(model, rank, pg, ddp_option, buffer_size)
    assert ddp_model is not None

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Container to hold: event -> list of events in milliseconds
    MODEL_FORWARD = "forward"
    MODEL_BACKWARD = "backward"
    metrics = {MODEL_FORWARD: [], MODEL_BACKWARD: []}

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} ...')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # TODO(bowangbj): Switch to real training set from ImageNet.
        inputs = torch.rand([32, 3, 224, 224], device=device)
        labels = torch.rand([32, 1000], device=device)

        # Forward
        start.record()
        outputs = ddp_model(inputs)
        end.record()
        torch.cuda.synchronize()
        metrics[MODEL_FORWARD].append(start.elapsed_time(end))

        # Backward
        start.record()
        loss_fn(outputs, labels).backward()
        # Reduce all grad, this is needed for non-DDP_CPP_CORE since the hook
        # for all_reduce does not exist yet.
        if ddp_option != DDPOption.DDP_CPP_CORE:
            ddp_model.all_reduce_grads()
        optimizer.step()
        optimizer.zero_grad()
        end.record()
        torch.cuda.synchronize()
        metrics[MODEL_BACKWARD].append(start.elapsed_time(end))

    if rank == 0:
        print(f'\nMetrics for GPU {rank} ddp_option {ddp_option}:')
        for step, elapsed_milliseconds in metrics.items():
            A = np.array(elapsed_milliseconds)
            print(' {N} iterations, {event}, mean={mean} ms, median={median} ms, p90={p90} ms, p99={p99} ms'.format(
                N=len(A), event=step, mean=np.mean(A),
                median=np.percentile(A, 50), p90=np.percentile(A, 90),
                p99=np.percentile(A, 99)))

def main():
    world_size = 2
    epochs = 1000
    # 4 MB. Note this is model dependent since # of buckets highly depends on
    # number of params and element number of params.
    buffer_size = 2 ** 22

    options = [
        DDPOption.DDP_CPP_CORE,
        DDPOption.PYTHON_DDP_ASYNC,
        DDPOption.PYTHON_DDP_SYNC,
        DDPOption.LEGACY_DISTRIBUTED_DATA_PARALLEL]

    for option in options:
        print("option: {}".format(option))
        mp.spawn(run_ddp,
                 args=(world_size, epochs, option, buffer_size),
                 nprocs=world_size,
                 join=True)

if __name__=="__main__":
    main()
