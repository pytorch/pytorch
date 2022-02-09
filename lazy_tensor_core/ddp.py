import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    try:
        # create model and move it to GPU with id rank
        # model = ToyModel().to(f"cuda:{rank}")
        model = ToyModel().to(f"lazy:{rank}")
        ddp_model = DDP(model, device_ids=[rank])
        # ddp_model = model

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        # outputs = ddp_model(torch.randn(20, 10))
        outputs = ddp_model(torch.randn(20, 10).to(f"lazy:{rank}"))
        # labels = torch.randn(20, 5).to(f"cuda:{rank}")
        labels = torch.randn(20, 5).to(f"lazy:{rank}")
        loss_fn(outputs, labels).backward()
        optimizer.step()
        print(f"Outputs: \n{outputs.to('cpu')}\n")
    except Exception as e:
        print(f"exception: {e}", flush=True)

    cleanup()


def run_demo(demo_fn, world_size):
    # it won't print exception messages in the child process.
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    print(f"main process id: {os.getpid()}")
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
