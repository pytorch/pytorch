import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    Replicate,
    Shard,
)

vocab_size = 32
embed_dim = 4
num_heads = 2
num_layers = 1
max_seq_len = 8
batch_size = 1
seq_len = 8
device = "cuda"
torch.manual_seed(0)

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tensor import DTensor, Replicate
if not dist.is_initialized():
    fake_store = FakeStore()
    dist.init_process_group(backend="fake", store=fake_store, rank=0, world_size=1)
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
device_mesh = init_device_mesh(
    "cuda",
    (world_size,),
    mesh_dim_names=("dp",),
)


class Transformer(nn.Module):
    def __init__(self, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh

    def forward(self, input_ids):
        input_ids = input_ids.redistribute(self.device_mesh, [Replicate()])
        input_ids = input_ids.to_local()
        return input_ids.sin()

model = Transformer(device_mesh).to(device)
compiled_model = torch.compile(model, fullgraph=True)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
input_ids_dt = DTensor.from_local(input_ids, device_mesh, [Shard(0)])
with FakeTensorMode(allow_non_fake_inputs=True):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    input_ids_dt = DTensor.from_local(input_ids, device_mesh, [Shard(0)])
    a = compiled_model(input_ids_dt)

