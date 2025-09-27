# mypy: allow-untyped-defs
import math
import traceback
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec


_compiled_autograd_enabled: bool = False


def detect_compiled_autograd():
    assert not torch.compiler.is_compiling(), (
        "`detect_compiled_autograd()` is designed to be called in eager mode"
    )
    global _compiled_autograd_enabled
    import torch._dynamo.compiled_autograd as ca

    _compiled_autograd_enabled = (
        ca.compiled_autograd_enabled
        or ca.compiled_autograd_enabled_force_eager
        or ca.in_compiled_autograd_region
    )


def compiled_autograd_enabled():
    global _compiled_autograd_enabled
    return _compiled_autograd_enabled


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


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = self.shard_process_group.rank()


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    def __post_init__(self):
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()


class TrainingState(Enum):
    """Describes the training state of one FSDP state / parameter group."""

    # Transition to forward starting pre-forward until post-forward
    FORWARD = auto()
    # Transition to pre-backward when unsharding in backward
    PRE_BACKWARD = auto()
    # Transition to post-backward when resharding and reducing gradients
    POST_BACKWARD = auto()
    # Idle before/after forward or before pre-backward/after post-backward
    IDLE = auto()


def _raise_assert_with_print(*args: Any, **kwargs: Any):
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)


def _is_composable_with_fsdp(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "replicate" not in registry


def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    return torch.Size([padded_dim0]) + tensor_size[1:]


def _chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> list[torch.Tensor]:
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


def _get_dim_chunked_size(
    chunk: torch.Tensor, unchunked_size: torch.Size, dim: int
) -> torch.Size:
    if chunk.numel() > 0:
        return chunk.size()
    # For 0 numel, we need to preserve nonzero-sized dims for DTensor APIs
    return unchunked_size[:dim] + torch.Size([0]) + unchunked_size[dim + 1 :]


def _from_local_no_grad(
    local_tensor: torch.Tensor,
    sharding_spec: DTensorSpec,
) -> DTensor:
    """
    This method is similar to ``DTensor.from_local()`` except that in eager mode
    it avoids some CPU overhead by avoiding default args and not being differentiable.
    """

    if not compiled_autograd_enabled():
        return DTensor(
            # Use the local tensor directly instead of constructing a new tensor
            # variable, e.g. with `view_as()`, since this is not differentiable
            local_tensor,
            sharding_spec,
            requires_grad=local_tensor.requires_grad,
        )
    else:
        return DTensor.from_local(
            local_tensor,
            sharding_spec.mesh,
            sharding_spec.placements,
            shape=sharding_spec.shape,
            stride=sharding_spec.stride,
        )


def _to_dtype_if_needed(
    tensor: torch.Tensor, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1
