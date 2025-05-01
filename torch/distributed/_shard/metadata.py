# mypy: allow-untyped-defs
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Union

from torch.distributed.remote_device import _remote_device


@dataclass
class ShardMetadata:
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
        bucket_id_offset: Optional[int] = None: This represents the bucket
        offset of the first bucket stored in this shard
        num_buckets: Optional[int] = None: This represents the number of
        buckets stored in this shard for bucket-wise sharding
    """

    __slots__ = ["shard_offsets", "shard_sizes", "placement", "bucket_id_offset", "num_buckets"]

    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: Optional[_remote_device]
    bucket_id_offset: Optional[int]
    num_buckets: Optional[int]

    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: Optional[Union[str, _remote_device]] = None,
        bucket_id_offset: Optional[int] = None,
        num_buckets: Optional[int] = None,
    ):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        self.bucket_id_offset = bucket_id_offset
        self.num_buckets = num_buckets
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")
        
        if self.bucket_id_offset:
            if self.bucket_id_offset < 0:
                raise ValueError("bucket_id_offset should be >=0 for all the shards")
            if not self.num_buckets:
                raise ValueError("num_buckets should be provided for bucket-wise sharding when bucket_offset is set")
            if self.num_buckets < 0:
                raise ValueError("Numebr of bucket should be > 0 within each shard when bucket-wise sharding is enabled")

    def __hash__(self):
        def _hash_reduce(a, b):
            return (a << 8) + hash(b)

        res = reduce(_hash_reduce, self.shard_offsets, 37)
        res = reduce(_hash_reduce, self.shard_sizes, res)
        if self.bucket_id_offset:
            res = reduce(_hash_reduce, self.bucket_id_offset, res)
            res = reduce(_hash_reduce, self.num_buckets, res)
        res = _hash_reduce(res, self.placement)
        return res
