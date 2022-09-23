from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor


class TensorFlattener:
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_tensor_flattener` to
    set a custom :class:`TensorFlattener` that implements the hooks.
    """

    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """E.g. converting ``DistributedTensor`` to local tensor."""
        ...

    def post_unflatten_transform(
        self,
        tensor: torch.Tensor,
        param_extension: Any,
    ) -> torch.Tensor:
        """E.g. converting local tensor to ``DistributedTensor``."""
        ...

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

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
        ...


_flattener: Optional[TensorFlattener] = None


def _set_tensor_flattener(flattener: TensorFlattener) -> None:
    global _flattener
    _flattener = flattener


def _tf_pre_flatten_transform(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Any]]:
    if _flattener is not None:
        new_tensor, extension = _flattener.pre_flatten_transform(tensor)
        if extension is not None:
            return new_tensor, extension
    return tensor, None


def _tf_post_unflatten_transform(
    tensor: torch.Tensor,
    param_extension: Any,
) -> torch.Tensor:
    if _flattener is not None and param_extension is not None:
        return _flattener.post_unflatten_transform(tensor, param_extension)
    return tensor


def _tf_chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    chunk_tensor_fn = (
        _flattener.chunk_tensor
        if _flattener is not None
        else _create_chunk_sharded_tensor
    )
    return chunk_tensor_fn(
        tensor,
        rank,
        world_size,
        num_devices_per_node,
        pg,
    )


def _tf_pre_load_state_dict_transform(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if _flattener is not None:
        return _flattener.pre_load_state_dict_transform(tensor)
    shards = tensor.local_shards()  # type: ignore[attr-defined]
    return (tensor, [shards[0].tensor] if shards else [])
