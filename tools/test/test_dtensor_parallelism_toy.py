# Run CMD:
# torchrun --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter 0,1 <filepath>.py

import os


LOG_INTERNAL = False
if LOG_INTERNAL:
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_LOGS"] = "+torch.distributed.tensor"
LOG_DEBUG = False
LOG_RANK = [0]

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._debug_mode import DebugMode

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    distribute_module,
    DTensor,
    Replicate,
    Shard,
    Partial,
)

def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    
    return rank, world_size, local_rank, device

class TinyNet(nn.Module):
    def __init__(self, d_in=8, d_hidden=16, d_out=1, mesh=None, grad_placements=None):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False) #8x16
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False) #16x1
        self.mesh = mesh
        self.grad_placements = grad_placements
        if self.mesh:
            print(f"[Rank {dist.get_rank()}]: ==== DTensor Mixed Local Ops Testing {grad_placements=} ====")
        else:
            print(f"[Rank {dist.get_rank()}]: ==== DTensor Testing ====")

    def forward(self, x):
        #x = 24x8 [12x8], [12x8]
        #w1 = 12x16 [12x8 @ 8x16 = 12x16 act]
        #w2 = 16x1 [12x16 @ 16x1 = 12x1 act]
        if not self.mesh:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

        w1_local = self.fc1.weight.to_local(grad_placements=self.grad_placements)
        x_local = x.to_local()
        x_local = F.linear(x_local, w1_local)
        x_local = F.relu(x_local)
        x2 = DTensor.from_local(x_local, device_mesh=self.mesh, placements=[Shard(0)])
        x2 = self.fc2(x2)
        return x2

def data_parallel_dtensor(local_ops=False, grad_placements=None):
    rank, world_size, local_rank, device = init_distributed()

    # 1) Data-parallel mesh across all ranks
    assert torch.cuda.is_available()
    mesh_device_type = "cuda"
    mesh = DeviceMesh(mesh_device_type, list(range(world_size)), mesh_dim_names=("dp",))

    # 2) Build model and distribute (replicate params for DP)
    torch.manual_seed(0)
    base_model = TinyNet(mesh=(mesh if local_ops else None), grad_placements=grad_placements).to(device)
    
    # 3) Global batch -> shard along batch dim across dp mesh
    global_batch_size = 24
    d_in = 8
    x_global = torch.randn(global_batch_size, d_in, device=device)
    y_global = torch.randn(global_batch_size, 1, device=device)

    # data parallelism
    with DebugMode(record_torchfunction=True) as debug_mode:
        model = distribute_module(base_model, mesh, partition_fn=lambda *args, **kwargs: Replicate())
        x = distribute_tensor(x_global, mesh, placements=[Shard(0)])
        y = distribute_tensor(y_global, mesh, placements=[Shard(0)])

    if LOG_DEBUG and rank in LOG_RANK:
        print(f"[Rank {dist.get_rank()}]: === DebugMode: Models/Input/Labels Sharding === \n {debug_mode.debug_string()}")

    # 4) Forward
    with DebugMode(record_torchfunction=True) as debug_mode:
        pred = model(x)  # DTensor with Shard(0)

    if LOG_DEBUG and rank in LOG_RANK:
        print(f"[Rank {dist.get_rank()}]: === DebugMode: Forward === \n {debug_mode.debug_string()}")

    # 5) MSE loss (pred - y)^2 .mean() yields a DTensor with a Partial state
    loss = (pred - y).pow(2).mean()

    if LOG_DEBUG:
        # full_tensor() triggers a collective call
        print(f"[Rank {dist.get_rank()}] Loss: {loss=} {loss.full_tensor()=} {loss.full_tensor().mean()=}")

    # 6) Backward + step — gradients on replicated params are auto all-reduced by DTensor autograd
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)

    with DebugMode(record_torchfunction=True) as debug_mode:
        loss.backward()

    if LOG_DEBUG and rank in LOG_RANK:
        print(f"[Rank {dist.get_rank()}]: === DebugMode: Loss backward === \n {debug_mode.debug_string()}")
    
    dist.barrier()

    with DebugMode(record_torchfunction=True) as debug_mode:
        opt.step()

    if LOG_DEBUG and rank in LOG_RANK:
        print(f"[Rank {dist.get_rank()}]: === DebugMode: Optimizer step === \n {debug_mode.debug_string()}")

    print(f"[Rank {dist.get_rank()}]: {local_ops=} {model.fc1.weight.norm()=} {model.fc2.weight.norm()=}")
    torch.testing.assert_close(model.fc1.weight.norm().to_local().item(), 2.326361656188965, msg=f"fc1 weights different from pure DTensor implementation {model.fc1.weight.norm()=}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # data_parallel_dtensor()
    # data_parallel_dtensor(local_ops=True)
    data_parallel_dtensor(local_ops=True, grad_placements=[Partial()])
