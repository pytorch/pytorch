r"""FlexShard: data-parallel sharding with pluggable placements (experimental).

``flex_shard`` shards a module's parameters, gradients and optimizer states
across a mesh, like :func:`~torch.distributed.fsdp.fully_shard`, but each bucket
of parameters names its own placement: how each rank stores its part and how
the bucket gathers parameters and reduces gradients. Parameters stay plain
``torch.Tensor`` local shards annotated with their sharding metadata (see
:func:`get_placements` and :func:`get_global_layout`); optimizers see ordinary
local tensors.

``fully_shard`` and ``flex_shard`` share one engine in
``torch.distributed.fsdp._flex_shard``: placements and bucket storage.
``flex_shard`` schedules its buckets with the engine's bucket runtime;
``fully_shard`` keeps one bucket per parameter group and schedules it per module.

.. warning::
    This API is experimental and may change without backward compatibility.
"""

from torch.distributed.fsdp._flex_shard import (
    BucketParamStorageLayout,
    BucketSpec,
    BucketStorageLayout,
    flex_shard,
    get_flex_shard_global_layouts,
    get_global_layout,
    get_global_shape,
    get_mesh,
    get_outer_layout,
    get_partial_grad_group,
    get_placements,
    GlobalLayout,
    GradientReduction,
    init_from_full,
    is_flex_shard_param,
    LocalStorageLayout,
    MixedPrecisionPolicy,
    OffloadPolicy,
    Placement,
    PlacementFn,
    register_optimizer_checkpoint_hook,
    relayout_bucket,
    set_global_layout,
    set_partial_grad_group,
    set_state_dict_global_layouts,
)
from torch.distributed.fsdp._flex_shard.placements import (
    BlockShard,
    BucketedBlockShard,
    BucketedOwned,
    BucketedOwnedSegmentSpec,
    make_bucketed_block_placement_fn,
    make_bucketed_owned_full_param_placement_fn,
    make_shard_placement_fn,
    MixedBucketPlacement,
    per_param_placements,
    Shard,
)


__all__ = [
    "BlockShard",
    "BucketedBlockShard",
    "BucketedOwned",
    "BucketedOwnedSegmentSpec",
    "BucketParamStorageLayout",
    "BucketSpec",
    "BucketStorageLayout",
    "flex_shard",
    "get_flex_shard_global_layouts",
    "get_global_layout",
    "get_global_shape",
    "get_mesh",
    "get_outer_layout",
    "get_partial_grad_group",
    "get_placements",
    "GlobalLayout",
    "GradientReduction",
    "init_from_full",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "make_bucketed_block_placement_fn",
    "make_bucketed_owned_full_param_placement_fn",
    "make_shard_placement_fn",
    "MixedBucketPlacement",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "per_param_placements",
    "Placement",
    "PlacementFn",
    "register_optimizer_checkpoint_hook",
    "relayout_bucket",
    "set_global_layout",
    "set_partial_grad_group",
    "set_state_dict_global_layouts",
    "Shard",
]
