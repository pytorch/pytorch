import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def broadcast(device):
    x = torch.zeros(20, 5).to(device)
    if device.index == 0:
        x = torch.randn(20, 5).to(device)
    dist.broadcast(x, 0)
    print(f"{os.getpid()} broadcast: {x.cpu()}")


def all_reduce(device):
    x = torch.full((20, 5), device.index + 1).to(device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"{os.getpid()} all_reduce: {x.cpu()}")


def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    device = torch.device("lazy", rank)
    # device = torch.device("cuda", rank)

    broadcast(device)
    all_reduce(device)

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
    # demo_basic(0, 1)
