import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore


# from torch.distributed.tensor.debug import visualize_sharding

# Use FakeTensorMode to handle CUDA tensors without actual CUDA
fake_mode = FakeTensorMode()

world_size = 4

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
device_mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (2, world_size // 2),
)

# Create fake CUDA tensor using FakeTensorMode
with fake_mode:
    x = torch.randn(1, 1, device="cuda")
    x = DTensor.from_local(x, device_mesh, [Shard(0), Shard(1)])
    print(x)
    # visualize_sharding(x)
    r = x.sum(1)
    print(r)
    # visualize_sharding(r)
