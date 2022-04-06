import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed import distributed_c10d
from torch.distributed.nn.functional import (
    reduce_scatter,
)

from .api import ShardedTensor


class _PartialTensor(object):
    """
    PartialTensor is an abstraction to represent Tensors that need
    aggregation across multiple devices and multiple processes.

    PartialTensor is initialized in an SPMD like fashion where each rank
    initializes the PartialTensor. The PartialTensor object on each rank
    then only stores the local partial shard, process group and the
    aggregation way to get a full tensor.

    PartialTensor doesn't provide any Tensor like operations but is a
    wrapper providing the Tensor representing the local partial shard.

    We assume the size of each local tensor to be exactly the same.

    Users can apply custom distributed sharded computations on top of
    this primitive.

    Args:
        local_partial_shard (Tensor): Partial result stored across ranks.
        process_group (ProcessGroup): The process group to aggregate on.
        reduce_op (distributed_c10d.ReduceOp): Way to aggregate the partial result.
            Default: ``distributed_c10d.ReduceOp.SUM``

    Examples:
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor = torch.cat([tensor, tensor + 2])
        >>> tensor
        tensor([1, 2, 3, 4]) # Rank 0
        tensor([3, 4, 5, 6]) # Rank 1
        >>> partial_tensor = _PartialTensor(tensor, distributed_c10d.ReduceOp.MAX)
        >>> sharding_dim = 0
        >>> collect_spec = shard_spec.ChunkShardingSpec(
                dim=sharding_dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                ],
            )
        >>> complete_tensor = partial_tensor.reshard(collect_spec)
        >>> complete_tensor
        ShardedTensor(
            ShardedTensorMetadata(
                shards_metadata=[
                    ShardMetadata(shard_offsets=[0], shard_sizes=[2], placement=rank:0/cuda:0),
                    ShardMetadata(shard_offsets=[2], shard_sizes=[2], placement=rank:1/cuda:1)],
                size=torch.Size([4])
        )
        >>> complete_tensor.local_tensor()
        tensor([3, 4]) # Rank 0
        tensor([5, 6]) # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1, 2]) + 2 * rank
        >>> tensor = torch.cat([tensor, tensor + 2])
        >>> tensor
        tensor([1, 2, 3, 4]) # Rank 0
        tensor([3, 4, 5, 6]) # Rank 1
        >>> partial_tensor = _PartialTensor(tensor)
        >>> complete_tensor = partial_tensor.reshard(collect_spec)
        >>> complete_tensor
        ShardedTensor(
            ShardedTensorMetadata(
                shards_metadata=[
                    ShardMetadata(shard_offsets=[0], shard_sizes=[2], placement=rank:0/cuda:0),
                    ShardMetadata(shard_offsets=[2], shard_sizes=[2], placement=rank:1/cuda:1)],
                size=torch.Size([4])
        )
        >>> complete_tensor.local_tensor()
        tensor([4, 6]) # Rank 0
        tensor([8, 10]) # Rank 1
    """

    def __init__(
        self, local_shard, process_group=None, reduce_op=distributed_c10d.ReduceOp.SUM
    ):
        self.local_shard = local_shard
        self.process_group = (
            process_group
            if process_group
            else dist.distributed_c10d._get_default_group()
        )
        self.reduce_op = reduce_op

    def __post_init__(self):
        if not isinstance(self.local_shard, torch.Tensor):
            raise ValueError("local_shard needs to be a Tensor.")
        if not isinstance(self.reduce_op, distributed_c10d.ReduceOp):
            raise ValueError(
                "reduce_op needs to be a member of distributed_c10d.ReduceOp."
            )

    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> ShardedTensor:
        """
        The reshard happens in two steps logically:

        1. Aggregate all the shards of the partial tensor.
        2. Shard this tensor according to the provided spec.

        In reality, for the sake of performance, we consolidate all partial tensors
        across multiple ranks and covert to a sharded tensor in one step.

        Args:
            resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
                The specification describing how we reshard the aggregated local result.

        Returns:
            A :class:`ShardedTensor` filled with local aggregated result.
        """
        if not isinstance(resharding_spec, shard_spec.ChunkShardingSpec):
            raise NotImplementedError("Only ChunkShardingSpec supported for reshard.")
        if self.local_shard.is_complex():
            raise NotImplementedError("Only real partial tensor supported for reshard.")
        sharding_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]
        chunk_mode_res = self.local_shard.size(sharding_dim) % self.process_group.size()
        # Add padding when the size is not divisible by the world size.
        if chunk_mode_res != 0:
            padding = [0] * (len(self.local_shard.size()) * 2)
            padding[-1] = self.process_group.size() - chunk_mode_res
            self.local_shard = torch.nn.functional.pad(
                self.local_shard,
                tuple(padding),
                "constant",
                0,
            )

        local_shards = self.local_shard.chunk(
            self.process_group.size(), dim=sharding_dim
        )
        local_result = reduce_scatter(
            torch.empty_like(local_shards[0]), list(local_shards), op=self.reduce_op
        )

        sharded_tensor_size = self.local_shard.size()
        return ShardedTensor._init_from_local_tensor(
            local_result,
            resharding_spec,
            sharded_tensor_size,
            process_group=self.process_group,
        )
