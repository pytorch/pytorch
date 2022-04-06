import os
import torch
import torch._lazy
import torch._lazy.ts_backend
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

torch._lazy.ts_backend.init()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.Backend.register_backend("lazy", torch._lazy.create_lazy_process_group)
    dist.init_process_group("lazy", rank=rank, world_size=world_size)


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

    # disable all JIT optimizations and fusions.
    torch._C._jit_set_bailout_depth(0)

    device = f"lazy:{rank}"

    # create model and move it to the device with id rank
    model = ToyModel().to(device)
    # Somehow, LazyTensor crashes if gradient_as_bucket_view is not set.
    # FIXME(alanwaketan): Investigate why and if we can remove this constraint.
    model = DDP(model, gradient_as_bucket_view=True)
    model.register_comm_hook(None, torch._lazy.lazy_comm_hook)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(101):
        optimizer.zero_grad()
        outputs = model(torch.randn(20, 10).to(device))
        labels = torch.randn(20, 5).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        torch._lazy.mark_step(device)

        if i % 10 == 0:
            print(f"{os.getpid()}: iteration {i}, loss {loss}")

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
