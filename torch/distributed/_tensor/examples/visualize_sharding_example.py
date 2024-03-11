import os

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor.debug.visualize_sharding import visualize_sharding

world_size = int(os.environ["WORLD_SIZE"])

# Example 1
tensor = torch.randn(4, 4)
mesh = DeviceMesh("cuda", list(range(world_size)))
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=1)])
if int(os.environ["LOCAL_RANK"]) == 0:
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
if int(os.environ["LOCAL_RANK"]) == 0:
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
if int(os.environ["LOCAL_RANK"]) == 0:
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
if int(os.environ["LOCAL_RANK"]) == 0:
    visualize_sharding(dtensor)
    """
             Col 0-3
    -------  ------------------
    Row 0-1  cuda:0, cuda:2
    Row 2-3  cuda:1, cuda:3
    """
