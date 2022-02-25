import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
# import lazy_tensor_core.debug.metrics as metrics

# from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-4'])

# from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook

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


def my_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    buffer = bucket.buffer()

    cuda_buffer = buffer.to(torch.device('cuda', buffer.device.index))
    torch.distributed.all_reduce(cuda_buffer)
    buffer.copy_(cuda_buffer)

    fut = torch.futures.Future()
    fut.set_result(buffer)

    return fut


def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    # disable all JIT optimizations and fusions.
    torch._C._jit_set_bailout_depth(0)

    # device = f"cuda:{rank}"
    device = f"lazy:{rank}"

    # create model and move it to GPU with id rank
    model = ToyModel().to(device)
    # we need to pass gradient_as_bucket_view=True as DDP internally sees gradients as
    # bucket views/alias because of LazyTensor. In more depth, the alias check relies on
    # TensorImpl's storages and LTCTensorImpl doesn't have storages so it confuses the
    # is_alias_of check to always return true.
    model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    # model.register_comm_hook(None, noop_hook)
    # model.register_comm_hook(None, my_hook)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(101):
        optimizer.zero_grad()
        outputs = model(torch.randn(20, 10).to(device))
        labels = torch.randn(20, 5).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        ltm.mark_step(device)

        if i % 10 == 0:
            print(f"{os.getpid()}: iteration {i}, loss {loss}")

    # print(metrics.metrics_report())

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
