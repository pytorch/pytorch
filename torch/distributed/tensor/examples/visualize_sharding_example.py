"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 visualize_sharding_example.py
"""

import os

import torch
import torch.distributed.tensor as dt
from torch.distributed.tensor.debug import visualize_sharding


world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

tensor = torch.ones((2, 2)) * rank

# Case 1: 1D mesh
mesh = dt.DeviceMesh("cuda", list(range(world_size)))
visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate()]))
visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate()]), use_rich=True)
visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0)]))
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0)]), use_rich=True
)
visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1)]))
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1)]), use_rich=True
)

# Case 2: 2D mesh
mesh = dt.DeviceMesh("cuda", [[0, 1], [2, 3]])
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Replicate()])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Replicate()]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Replicate()])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Replicate()]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=1)])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=1)]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Replicate()])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Replicate()]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=0)])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=0)]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Shard(dim=1)])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Shard(dim=1)]),
    use_rich=True,
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Shard(dim=0)])
)
visualize_sharding(
    dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Shard(dim=0)]),
    use_rich=True,
)
