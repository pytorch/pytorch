from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)


class FSDPExtensions(ABC):
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """

    @abstractmethod
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """E.g. converting ``DistributedTensor`` to local tensor."""
        ...

    @abstractmethod
    def post_unflatten_transform(
        self,
        tensor: torch.Tensor,
        param_extension: Any,
    ) -> torch.Tensor:
        """E.g. converting local tensor to ``DistributedTensor``."""
        ...

    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Shards a tensor to chunks and returns the local chunk."""
        ...

    @abstractmethod
    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Shard]]:
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
        ...


_extensions: Optional[FSDPExtensions] = None


def _set_fsdp_extensions(flattener: FSDPExtensions) -> None:
    global _extensions
    _extensions = flattener


def _ext_pre_flatten_transform(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Any]]:
    if _extensions is not None:
        new_tensor, extension = _extensions.pre_flatten_transform(tensor)
        if extension is not None:
            return new_tensor, extension
    return tensor, None


def _ext_post_unflatten_transform(
    tensor: torch.Tensor,
    param_extension: Any,
) -> torch.Tensor:
    if _extensions is not None and param_extension is not None:
        return _extensions.post_unflatten_transform(tensor, param_extension)
    return tensor


def _ext_chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    chunk_tensor_fn = (
        _extensions.chunk_tensor
        if _extensions is not None
        else _create_chunk_sharded_tensor
    )
    return chunk_tensor_fn(
        tensor,
        rank,
        world_size,
        num_devices_per_node,
        pg,
    )


def _ext_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> torch.Tensor:
    # TODO: Address composability issue and remove the assertion.
    assert (
        _extensions is None
    ), "Currently does not support composability when _use_dtensor = True"
    return _create_chunk_dtensor(
        tensor,
        rank,
        device_mesh,
    )


def _ext_pre_load_state_dict_transform(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[Shard]]:
    if _extensions is not None:
        return _extensions.pre_load_state_dict_transform(tensor)

    assert type(tensor) is ShardedTensor
    shards = tensor.local_shards()
    return (tensor, shards)
