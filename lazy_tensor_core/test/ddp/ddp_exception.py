import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.Backend.register_backend("lazy", ltm.create_lazy_process_group)
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

    model = ToyModel().to(f"lazy:{rank}")
    ddp_model = DDP(model, gradient_as_bucket_view=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(2):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(f"lazy:{rank}"))
        labels = torch.randn(20, 5).to(f"lazy:{rank}")
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"loss: \n{loss}")

    cleanup()

# This test is used to debug exceptions in a single process setup.
if __name__ == "__main__":
    print(f"main process id: {os.getpid()}")
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    demo_basic(0, world_size=1)
