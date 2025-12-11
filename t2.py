import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import ExitStack
from functools import partial
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
import torch
import torch.nn as nn
import torch
import sys
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DTensor

class Repro(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, device_mesh):
        y = x.to_local()
        y = y.sin()
        y = DTensor.from_local(
            y, device_mesh, [Replicate(), Replicate(), Replicate()]
        )
        y = y.redistribute(device_mesh, [Shard(0), Replicate(), Replicate()])
        return y

world_size = 128
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (8, 2, 2),
    mesh_dim_names=(
        "dim1", "dim2", "dim3",
    ),
)
placements = (Replicate(), Replicate(), Replicate())

arg0 = torch.rand([5, 3], dtype=torch.float16, device='cuda', requires_grad=True) # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
arg0 = DTensor.from_local(arg0, mesh, placements)

mod = Repro()
mod = torch.compile(mod)

out = mod(arg0, mesh)
loss = out.mean()
loss.backward()

