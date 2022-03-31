import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.Backend.register_backend("lazy", ltm.create_lazy_process_group)
    dist.init_process_group("lazy", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def broadcast(device):
    x = torch.zeros(2, 3).to(device)
    if device.index == 0:
        x = torch.randn(2, 3).to(device)
    dist.broadcast(x, 0)
    print(f"{os.getpid()} broadcast: {x.cpu()}")


def all_reduce(device):
    x = torch.full((2, 3), device.index + 1).to(device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"{os.getpid()} all_reduce: {x.cpu()}")

def all_gather(device):
    tensor_list = [torch.zeros(2, 3, dtype=torch.int64) for _ in range(2)]
    x = torch.full((2, 3), device.index + 1).to(device)
    dist.all_gather(tensor_list, x)
    print(f"{os.getpid()} all_gather: {[tensor.cpu() for tensor in tensor_list]}")


def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    device = torch.device("lazy", rank)

    broadcast(device)
    all_reduce(device)
    all_gather(device)

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
