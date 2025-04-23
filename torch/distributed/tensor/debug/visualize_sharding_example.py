# OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 visualize_sharding_example.py
import contextlib
import importlib.util
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing.errors
import torch.distributed.tensor as dt
from torch._prims_common import ShapeType

from _visualize_sharding import visualize_sharding


@contextlib.contextmanager
def distributed_context():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        local_device = torch.device("cpu", local_rank)
        dist.init_process_group(backend="gloo")
        yield local_device
    finally:
        dist.barrier()
        dist.destroy_process_group()
        print(f"Rank {local_rank} finished")


@torch.distributed.elastic.multiprocessing.errors.record
def main(local_device):
    local_rank = local_device.index
    tensor = torch.ones((2, 2), device=local_device) * local_rank

    # Case 1: 1D mesh
    mesh = dist.init_device_mesh("cpu", (4,), mesh_dim_names=["dp"])
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate()]))
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate()]), use_rich=True)
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0)]))
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0)]), use_rich=True)
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1)]))
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1)]), use_rich=True)

    # Case 2: 2D mesh
    mesh = dist.init_device_mesh("cpu", (2, 2), mesh_dim_names=["dp", "tp"])
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Replicate()]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Replicate()]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Replicate()]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Replicate()]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=1)]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=1)]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Replicate()]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Replicate()]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=0)]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Replicate(), dt.Shard(dim=0)]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Shard(dim=1)]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=0), dt.Shard(dim=1)]), use_rich=True
    )
    visualize_sharding(dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Shard(dim=0)]))
    visualize_sharding(
        dt.DTensor.from_local(tensor, mesh, [dt.Shard(dim=1), dt.Shard(dim=0)]), use_rich=True
    )


if __name__ == "__main__":
    with distributed_context() as local_device:
        main(local_device)
