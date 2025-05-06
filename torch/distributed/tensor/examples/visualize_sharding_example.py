"""
To run the example, use the following command:
TERM=xterm-256color torchrun --nproc-per-node=4 visualize_sharding_example.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.tensor as dt
from torch.distributed.tensor.debug import visualize_sharding
import rich
import rich.rule


assert int(os.getenv("WORLD_SIZE", "1")) >= 4, "We need at least 4 devices"
rank = int(os.environ["RANK"])


def section(msg: str):
    if rank == 0:
        rich.print(rich.rule.Rule(msg))


def visualize(t, msg: str = ""):
    if rank == 0:
        rich.print(msg)
        dt.debug.visualize_sharding(t, use_rich=False)
        dt.debug.visualize_sharding(t, use_rich=True)


section(f"[bold]1D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh("cuda", (4,))
t = torch.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard along the only tensor dimension",
)

section(f"[bold]2D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh("cuda", (4,))
t = torch.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard alone the first tensor dimension along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1)]),
    "Shard along the second tensor dimension along the only mesh dimension",
)

section(f"[bold]1D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh("cuda", (2, 2))
t = torch.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the only tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along the second mesh dimension",
)

section(f"[bold]2D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh("cuda", (2, 2))
t = torch.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=1)]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Replicate()]),
    "Shard the second tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)


dist.destroy_process_group()
