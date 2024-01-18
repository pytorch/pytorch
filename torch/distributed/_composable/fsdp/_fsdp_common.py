import traceback

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, cast, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed._tensor import DeviceMesh, DTensor, Placement


FSDP_SHARDED = "_fsdp_sharded"
FSDP_IGNORED = "_fsdp_ignored"
FSDP_ENABLE_LOGGING = False


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )
        self.mesh_size: int = self.mesh.size()


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = cast(
            dist.ProcessGroup, self.mesh.get_group(self.shard_mesh_dim)
        )
        self.shard_mesh_rank: int = self.shard_process_group.rank()


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = cast(
            dist.ProcessGroup, self.mesh.get_group(self.replicate_mesh_dim)
        )
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    def __post_init__(self):
        super(FSDPMeshInfo, self).__post_init__()
        super(DDPMeshInfo, self).__post_init__()


class TrainingState(Enum):
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()
    IDLE = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``.

    We additionally save shared modules and shared parameter names to update
    them accordingly.
    """

    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


class FSDPInternalError(AssertionError):
    pass


def print_and_raise_internal(*args: Any, **kwargs: Any):
    print(f"[Rank {torch.distributed.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise FSDPInternalError(*args, **kwargs)


def _is_composable_with_fsdp(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    # TODO: Add the TorchRec composable API name.
    return "replicate" not in registry


def _normalize_device(device: Union[torch.device, int, str]) -> torch.device:
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


def _cast_floating_point_tensor(
    dtype: torch.dtype,
    x: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)


def get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    if tensor_size[0] < dim0_factor:
        padded_size = torch.Size([dim0_factor]) + tensor_size[1:]
    elif tensor_size[0] % dim0_factor != 0:
        padded_size = (
            torch.Size([tensor_size[0] + dim0_factor - (tensor_size[0] % dim0_factor)])
            + tensor_size[1:]
        )
    else:
        padded_size = tensor_size
    return cast(torch.Size, padded_size)


def from_local_no_grad(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
    global_size: torch.Size,
    global_stride: Tuple[int, ...],
) -> DTensor:
    """
    This method is similar to ``DTensor.from_local()`` except it avoids some
    CPU overhead by avoiding default args and not being differentiable.
    """
    return DTensor(
        # Use the local tensor directly instead of constructing a new tensor
        # variable, e.g. with `view_as()`, since this is not differentiable
        local_tensor,
        device_mesh,
        placements,
        shape=global_size,
        dtype=local_tensor.dtype,
        requires_grad=local_tensor.requires_grad,
        stride=global_stride,
    )


def chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> List[torch.Tensor]:
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


def to_dtype_if_needed(
    tensor: torch.Tensor, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor
