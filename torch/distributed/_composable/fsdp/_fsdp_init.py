import torch
import torch.distributed as dist

from torch._prims_common import DeviceLikeType

from torch.distributed._tensor import DeviceMesh, init_device_mesh


def _normalize_device(device: DeviceLikeType) -> torch.device:
    if isinstance(device, torch.device):
        if device == torch.device("cuda"):
            return torch.device("cuda", torch.cuda.current_device())
        return device
    elif isinstance(device, int):
        return torch.device("cuda", device)
    elif isinstance(device, str):
        if device == "cuda":
            return torch.device(device, torch.cuda.current_device())
        return torch.device(device)
    else:
        raise TypeError(f"Invalid type for device {device}: {type(device)}")


def _init_default_fully_shard_mesh(device_type: str) -> DeviceMesh:
    """The default fully-shard mesh shards over the global mesh."""
    default_pg = dist.distributed_c10d._get_default_group()
    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(default_pg.size(),),
        mesh_dim_names=("dp_shard",),
    )
    return mesh
