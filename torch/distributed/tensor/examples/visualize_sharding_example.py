"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 visualize_sharding_example.py
"""

import os

import torch
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed.tensor.debug import visualize_sharding


world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

# Example 1
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", list(range(world_size)))
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=1)])
visualize_sharding(dtensor)
"""
            Col 0-0    Col 1-1    Col 2-2    Col 3-3
-------  ---------  ---------  ---------  ---------
Row 0-3  cuda:0   cuda:1   cuda:2   cuda:3
"""

# Example 2
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", list(range(world_size)))
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0)])
visualize_sharding(dtensor)
"""
            Col 0-3
-------  ---------
Row 0-0  cuda:0
Row 1-1  cuda:1
Row 2-2  cuda:2
Row 3-3  cuda:3
"""

# Example 3
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0), Replicate()])
visualize_sharding(dtensor)
"""
            Col 0-3
-------  ------------------
Row 0-1  cuda:0, cuda:1
Row 2-3  cuda:2, cuda:3
"""

# Example 4
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
dtensor = distribute_tensor(tensor, mesh, [Replicate(), Shard(dim=0)])
visualize_sharding(dtensor)
"""
            Col 0-3
-------  ------------------
Row 0-1  cuda:0, cuda:2
Row 2-3  cuda:1, cuda:3
"""

# Example 5: single-rank submesh
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", [rank])
dtensor = distribute_tensor(tensor, mesh, [Replicate()])
visualize_sharding(dtensor, header=f"Example 5 rank {rank}:")
"""
Example 5 rank 0:
         Col 0-3
-------  ---------
Row 0-3  cuda:0

Example 5 rank 1:
         Col 0-3
-------  ---------
Row 0-3  cuda:1

Example 5 rank 2:
         Col 0-3
-------  ---------
Row 0-3  cuda:2

Example 5 rank 3:
         Col 0-3
-------  ---------
Row 0-3  cuda:3
"""
