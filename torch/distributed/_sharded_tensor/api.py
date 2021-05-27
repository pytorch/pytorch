from typing import List

import torch
import torch.distributed as dist
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    ShardMetadata,
    ShardingSpec,
)
from torch.distributed._sharding_spec._internals import is_valid_device
from torch.distributed.utils import _parse_remote_device


class Shard(object):
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.
    """
    __slots__ = ['_tensor', '_metadata']

    def __init__(self, tensor: torch.Tensor, metadata: ShardMetadata):
        self._tensor = tensor
        self._metadata = metadata

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def metadata(self) -> ShardMetadata:
        return self._metadata


class ShardedTensor(object):
    """
    ShardedTensor is an abstraction to represent Tensors that are sharded
    across multiple devices and multiple processes.

    ShardedTensor is initialized in an SPMD like fashion where each rank
    initializes the ShardedTensor. The ShardedTensor object on each rank
    then only stores the local shard for the Tensor and provides global
    metadata for all the shards.

    ShardedTensor doesn't provide any Tensor like operations but is a wrapper
    providing the Tensor representing the local shard and the global metadata.
    Using these, users can build their custom distributed sharded computations
    on top of this primitive. The local shards are all initialized using
    :meth:`torch.empty`.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. If specified the ShardedTensor is only
            built on ranks that are part of this process group and the provided ``sharding_spec``
            is applied in the context of this process group.
    """

    def __init__(
        self,
        sharding_spec: ShardingSpec,
        *size,
        dtype=None,
        layout=torch.strided,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
        process_group=None,
    ):
        if layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')

        if memory_format != torch.contiguous_format:
            raise ValueError('Only torch.contiguous_format memory_format is currently supported')

        self._sharding_spec = sharding_spec
        self._dims = list(size)
        self._process_group = (
            process_group
            if process_group is not None
            else torch.distributed.distributed_c10d._get_default_group()
        )

        if torch.distributed.distributed_c10d._rank_not_in_group(self._process_group):
            raise ValueError(f'Global rank: {dist.get_rank()} not part of process group')

        self._local_shards: List[Shard] = []
        self._sharding_metadata: List[ShardMetadata] = []
        if isinstance(self._sharding_spec, ChunkShardingSpec):
            self._init_chunked(
                self._sharding_spec,
                self._dims,
                dtype,
                layout,
                requires_grad,
                pin_memory,
                memory_format,
                process_group,
            )

    def _init_chunked(
        self,
        sharding_spec: ChunkShardingSpec,
        dims,
        dtype,
        layout,
        requires_grad,
        pin_memory,
        memory_format,
        process_group,
    ):
        current_rank = dist.get_rank(process_group)
        sharding_dim = sharding_spec.dim

        # Validate the sharding spec.
        if not isinstance(sharding_dim, int):
            raise ValueError(
                f"Sharding dim needs to be an integer, found: {sharding_dim}"
            )
        if sharding_dim >= len(dims) or sharding_dim < -len(dims):
            raise ValueError(f"Invalid sharding dim: {sharding_dim}")

        dim_size = dims[sharding_dim]
        devices = sharding_spec.placements
        chunks = len(devices)
        # split_size computed similar to 'torch.chunk'
        split_size = (dim_size + chunks - 1) // chunks

        for idx, device in enumerate(devices):
            if not is_valid_device(device):
                raise ValueError(f"{device} is not a valid device")

            rank, local_device = _parse_remote_device(device)  # type: ignore[arg-type]

            # Validate rank.
            if not isinstance(rank, int) or (rank < 0 or rank >= dist.get_world_size(process_group)):
                raise ValueError(f'Invalid rank: {rank}')

            # Adjust the sharding dim for this rank.
            sharded_dim_size = min(dim_size, split_size * (idx + 1)) - split_size * idx

            if sharded_dim_size > 0:
                # Build sharding_metadata.

                # deepcopy for modification.
                rank_dims = dims.copy()

                rank_offsets = [0] * len(dims)
                rank_offsets[sharding_dim] = split_size * idx
                rank_dims[sharding_dim] = sharded_dim_size

                shard_metadata = ShardMetadata(rank_offsets, rank_dims, device)
                self._sharding_metadata.append(shard_metadata)

                # Build the local shard for the current rank if it is involved in the sharding spec.
                if current_rank == rank:
                    # Initialize the local shard.
                    local_shard = torch.empty(
                        *rank_dims,
                        dtype=dtype,
                        layout=layout,
                        device=local_device,
                        requires_grad=requires_grad,
                        memory_format=memory_format,
                        pin_memory=pin_memory,
                    )

                    self._local_shards.append(Shard(local_shard, shard_metadata))

    def sharding_spec(self) -> ShardingSpec:
        """
        Returns the ShardingSpec for the tensor.
        """
        return self._sharding_spec

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(f"torch function '{func.__name__}' not supported for ShardedTensor!")

    def sharding_metadata(self) -> List[ShardMetadata]:
        """
        Returns a list of :class:`ShardeMetadata` objects corresponding to the
        metadata for each shard.
        """
        return self._sharding_metadata

    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards
