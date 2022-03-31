import copy
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

# from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-4'])

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


def step(model, input, labels, device, loss_fn, optimizer):
    input = input.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    output = model(input)

    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    return loss


def all_close(parameters_a, parameters_b):
    for param_a, param_b in zip(parameters_a, parameters_b):
        # The precision is quite low, which makes me consider this might not
        # be a good idea. We probably should just use the old fashion of observing
        # the learning curve of the two to determine if they converge at the same rate.
        if not torch.allclose(param_a.cpu(), param_b.cpu(), atol=1e-01):
            print_param(param_a, param_b)
            return False
    return True


def print_param(param_a, param_b):
    print("=====lazy=====")
    print(param_a.cpu())
    print("=====cuda=====")
    print(param_b.cpu())

def init(model, device, rank, size):
    model = copy.deepcopy(model).to(device)
    if device.type == "lazy":
        model = DDP(model, gradient_as_bucket_view=True)
    if device.type == "cuda":
        model = DDP(
            model,
            gradient_as_bucket_view=True,
            device_ids=[rank],
            process_group=dist.ProcessGroupNCCL(
                dist.distributed_c10d._get_default_store(), rank, size, timedelta(minutes=3))
        )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    torch._C._jit_set_bailout_depth(0)

    model = ToyModel()

    device_lazy = torch.device("lazy", rank)
    model_lazy, loss_fn_lazy, optimizer_lazy = init(model, device_lazy, rank, world_size)

    device_cuda = torch.device("cuda", rank)
    model_cuda, loss_fn_cuda, optimizer_cuda = init(model, device_cuda, rank, world_size)

    assert all_close(model_lazy.parameters(), model_cuda.parameters())
    for i in range(101):
        input = torch.randn(20, 10)
        labels = torch.randn(20, 5)

        loss_lazy = step(model_lazy, input, labels, device_lazy, loss_fn_lazy, optimizer_lazy)
        loss_cuda = step(model_cuda, input, labels, device_cuda, loss_fn_cuda, optimizer_cuda)

        if not all_close(model_lazy.parameters(), model_cuda.parameters()):
            break

        if i % 10 == 0:
            print(f"{os.getpid()}: iteration {i} lazy_parameters ~= cuda_parameters, loss_lazy={loss_lazy}, loss_cuda={loss_cuda}")

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
