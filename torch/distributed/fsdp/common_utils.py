"""
Public common utilities for FSDP.
"""

from enum import auto, Enum


class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.

    - ``FULL_SHARD``: Parameters, gradients, and optimizer states are sharded.
    For the parameters, this strategy unshards (via all-gather) before the
    forward, reshards after the forward, unshards before the backward
    computation, and reshards after the backward computation. For gradients, it
    synchronizes and shards them (via reduce-scatter) after the backward
    computation. The sharded optimizer states are updated locally per rank.
    - ``SHARD_GRAD_OP``: Gradients and optimizer states are sharded during
    computation, and additionally, parameters are sharded outside computation.
    For the parameters, this strategy unshards before the forward, does not
    reshard them after the forward, and only reshards them after the backward
    computation. The sharded optimizer states are updated locally per rank.
    Inside ``no_sync()``, the parameters are not resharded after the backward
    computation.
    - ``NO_SHARD``: Parameters, gradients, and optimizer states are not sharded
    but instead replicated across ranks similar to PyTorch's
    :class:`DistributedDataParallel` API. For gradients, this strategy
    synchronizes them (via all-reduce) after the backward computation. The
    unsharded optimizer states are updated locally per rank.
    """

    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    # HYBRID_SHARD = auto()


class BackwardPrefetch(Enum):
    """
    This configures explicit backward prefetching. For NCCL backend, any
    collectives, even if issued in different streams, contend for the same
    per-device NCCL stream, which is why the relative order in which the
    collectives are issued matters for overlapping. The different backward
    prefetching settings correspond to different orderings.

    - ``BACKWARD_PRE``: This prefetches the next set of parameters before the
    current set of parameter's gradient computation. This improves backward
    pass throughput by overlapping communication (next all-gather) and
    computation (current gradient computation) but may increase the peak memory
    usage since two sets of parameters are in memory at once.
    - ``BACKWARD_POST``: This prefetches the next set of parameters after the
    current set of parameter's gradient computation. This may improve backward
    pass throughput by overlapping communication (current reduce-scatter) and
    computation (next gradient computation) and does not increase the peak
    memory usage since the current set of parameters are freed before
    prefetching the next set. Specifically, the next all-gather is reordered to
    be before the current reduce-scatter.

    For both modes, the ordering that defines "current" and "next" is not
    always correct in the current implementation, so this may cause some
    performance regression for some models.
    """

    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
