# mypy: allow-untyped-defs
import functools
import math
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec

from .._flex_shard.utils import _cast_fp_tensor, TrainingState  # noqa: F401
from ._fsdp_api import DataParallelMeshDims


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _dynamo_disable(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Disable dynamo tracing for FSDP hooks."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs):
        return torch._dynamo.disable(
            func, recursive=True, reason="skipping FSDP hooks"
        )(*args, **kwargs)

    return wrapper


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: int | None = None
    replicate_mesh_dim: int | None = None
    dp_mesh_dims: DataParallelMeshDims | None = None
    # The full SPMD mesh (excluding PP dims) that params are distributed on.
    # Must include all non-PP SPMD dims (e.g. DP + TP); passing a submesh
    # that omits dims like TP will lead to incorrect behavior.
    spmd_mesh: DeviceMesh | None = field(default=None, repr=False)
    is_spmd_mesh: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )
        self.is_spmd_mesh = self.dp_mesh_dims is not None

    @property
    def shard_info(self) -> "FSDPMeshInfo | None":
        """This mesh info if it shards parameters (FSDP, HSDP), else None."""
        return None

    @property
    def replicate_info(self) -> "DDPMeshInfo | None":
        """This mesh info if it replicates parameters (HSDP, DDP), else None."""
        return None


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = self.shard_process_group.rank()

    @property
    def shard_info(self) -> "FSDPMeshInfo | None":
        return self


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()

    @property
    def replicate_info(self) -> "DDPMeshInfo | None":
        return self


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    def __post_init__(self):  # pylint:disable=useless-parent-delegation
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()


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


# Unused within FSDP since the reduce-scatter buffer is sized from the padded
# sharded parameters; kept for external callers that import it from here.
def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    return torch.Size([padded_dim0]) + tensor_size[1:]


def _from_local_no_grad(
    local_tensor: torch.Tensor,
    sharding_spec: DTensorSpec,
) -> DTensor:
    """
    This method is similar to ``DTensor.from_local()`` except that in eager mode
    it avoids some CPU overhead by avoiding default args and not being differentiable.
    """
    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # Use the local tensor directly instead of constructing a new tensor
        # variable, e.g. with `view_as()`, since this is not differentiable
        # pyrefly: ignore [bad-argument-count]
        local_tensor,
        sharding_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=local_tensor.requires_grad,
    )


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1


@dataclass
class ShardPlacementResult:
    placement: Shard | None
    mesh_info: FSDPMeshInfo


ShardPlacementFnResult = Shard | ShardPlacementResult | None


def resolve_shard_placement(
    result: ShardPlacementFnResult,
    default_mesh_info: FSDPMeshInfo,
) -> ShardPlacementResult:
    """Resolve the shard_placement_fn result to a ShardPlacementResult.

    Handles different input types and applies defaults:
    - None: Use default sharding (Shard(0)) on default mesh
    - Shard: Use specified shard dimension on default mesh
    - ShardPlacementResult: Use as-is

    Args:
        result: The return value from shard_placement_fn, or None if no fn provided.
        default_mesh_info: The default FSDPMeshInfo to use if not specified.

    Returns:
        A ShardPlacementResult with placement and mesh_info.
    """
    if result is None:
        return ShardPlacementResult(placement=None, mesh_info=default_mesh_info)
    if isinstance(result, Shard):
        return ShardPlacementResult(placement=result, mesh_info=default_mesh_info)
    if isinstance(result, ShardPlacementResult):
        return result
    raise ValueError(f"Invalid shard_placement_fn result: {result}")
