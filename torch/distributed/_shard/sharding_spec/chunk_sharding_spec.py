from dataclasses import dataclass
import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device
)
from torch.distributed._shard._utils import narrow_tensor
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from typing import List, Union, TYPE_CHECKING
from ._internals import (
    get_chunked_dim_size,
    get_split_size,
)

from .api import ShardingSpec

if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor

@dataclass
class ChunkShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    The semantics of how a tensor is partitioned is inline with
    :meth:`torch.chunk`, where ``dim`` in torch.chunk corresponds to the
    specified ``dim`` and ``chunks`` in torch.chunk is the number of elements
    in the placement specified.

    Args:
        dim (int or str):
            The dimension to shard on, could be an integer representing the
            dimension or a string in case of named tensors where dimensions are
            named. Note that named tensor support is not added yet.
        placement(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
    """

    ShardingDim = Union[int, str]

    dim: ShardingDim
    placements: List[Union[torch.distributed._remote_device, str]]

    def __post_init__(self):
        self._verify_dim(self.dim)
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    @staticmethod
    def _verify_dim(dim):
        # Validate the sharding spec.
        # TODO: support named dimension
        if isinstance(dim, str):
            raise NotImplementedError(
                "ChunkShardingSpec does not support named dimension yet!"
            )

        if not isinstance(dim, int):
            raise ValueError(
                f"Sharding dim needs to be an integer, found: {dim}"
            )

    def build_metadata(self,
                       tensor_sizes: torch.Size,
                       tensor_properties: sharded_tensor_meta.TensorProperties,
                       ) -> sharded_tensor_meta.ShardedTensorMetadata:
        tensor_num_dim = len(tensor_sizes)

        self._verify_dim(self.dim)
        if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
            raise ValueError(f"Invalid sharding dim: {self.dim}")

        shards_metadata = []
        sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
        chunks = len(self.placements)
        split_size = get_split_size(sharding_dim_size, chunks)
        for idx, placement in enumerate(self.placements):
            # generate ShardMetadata for each placement device
            chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
            shard_size = list(tensor_sizes)
            current_offsets = [0] * tensor_num_dim
            current_offsets[self.dim] = split_size * idx  # type: ignore[index]
            shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

            shard_metadata = ShardMetadata(
                shard_offsets=current_offsets,
                shard_sizes=shard_size,
                placement=placement,
            )
            shards_metadata.append(shard_metadata)

        return sharded_tensor_meta.ShardedTensorMetadata(
            shards_metadata,
            tensor_sizes,
            tensor_properties
        )


    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> "ShardedTensor":
        """
        Args:
            src_rank: group rank relative to ``process_group``

            N.B. If ``process_group`` is None, ``src_rank`` is a global rank.
        """
        # relative imports to avoid circular dependency
        from torch.distributed._shard.sharded_tensor import (
            ShardedTensor
        )
        tensor_properties = sharded_tensor_meta.TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned()
        )
        current_rank = dist.get_rank(process_group)
        tensor_meta = self.build_metadata(tensor.size(), tensor_properties)
        local_shards = []
        local_tensor = None
        local_metadata = None
        tensors_to_scatter = [None] * dist.get_world_size(process_group)

        sharding_dim_size = tensor.size()[self.dim]  # type: ignore[index]
        chunks = len(self.placements)
        split_size = get_split_size(sharding_dim_size, chunks)
        scatter_shape = list(tensor.size())
        scatter_shape[self.dim] = split_size  # type: ignore[index]

        for shard_meta in tensor_meta.shards_metadata:
            rank, device = _parse_and_validate_remote_device(process_group, shard_meta.placement)
            if current_rank == src_rank:
                # Reshape to get shard for this rank and we don't want autograd
                # recording here for the narrow op and 'local_shard' should be a
                # leaf variable in the autograd graph.
                narrowed_tensor = narrow_tensor(tensor, shard_meta)
                if shard_meta.shard_sizes[self.dim] < split_size:  # type: ignore[index]
                    # for the last shard that might be smaller to other shards
                    # resize the narrowed tensor to the same size and use it for
                    # the scatter collective as dist.scatter requires same size
                    # inputs on every rank
                    tensor_to_scatter = narrowed_tensor.detach().clone().resize_(scatter_shape)
                else:
                    tensor_to_scatter = narrowed_tensor.detach().clone().contiguous()

                tensors_to_scatter[rank] = tensor_to_scatter

            if current_rank == rank:
                local_tensor = torch.empty(
                    scatter_shape, dtype=tensor.dtype, layout=tensor.layout, device=device)
                local_metadata = shard_meta

        # each rank should have local_tensor and local_metadata initialized if we build
        # the metadata list in a correct way.
        assert local_tensor is not None
        assert local_metadata is not None

        # Scatter the shards to all ranks in the pg
        # scatter takes the global rank as ``src``
        src_for_scatter = src_rank
        if process_group is not None and process_group is not distributed_c10d._get_default_group():
            src_for_scatter = distributed_c10d.get_global_rank(process_group, src_for_scatter)

        dist.scatter(
            local_tensor,
            scatter_list=tensors_to_scatter if current_rank == src_rank else None,
            src=src_for_scatter,
            group=process_group
        )

        if list(local_tensor.size()) != local_metadata.shard_sizes:
            # detach again after receiving to ensure local shards remain a leaf node
            local_tensor = local_tensor.resize_(local_metadata.shard_sizes).detach()

        # Sync requires_grad to local_shard.
        local_tensor.requires_grad = tensor.requires_grad

        local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards,
            tensor_meta,
            process_group=process_group)

        # Manually set sharding_spec
        st._sharding_spec = self

        return st
