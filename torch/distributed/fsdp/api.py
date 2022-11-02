"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

from dataclasses import dataclass
from enum import auto, Enum

from typing import Optional

import torch

__all__ = ["ShardingStrategy", "BackwardPrefetch", "MixedPrecision", "CPUOffload"]


class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.

    - ``FULL_SHARD``: Parameters, gradients, and optimizer states are sharded.
      For the parameters, this strategy unshards (via all-gather) before the
      forward, reshards after the forward, unshards before the backward
      computation, and reshards after the backward computation. For gradients,
      it synchronizes and shards them (via reduce-scatter) after the backward
      computation. The sharded optimizer states are updated locally per rank.
    - ``SHARD_GRAD_OP``: Gradients and optimizer states are sharded during
      computation, and additionally, parameters are sharded outside
      computation. For the parameters, this strategy unshards before the
      forward, does not reshard them after the forward, and only reshards them
      after the backward computation. The sharded optimizer states are updated
      locally per rank. Inside ``no_sync()``, the parameters are not resharded
      after the backward computation.
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
    This configures explicit backward prefetching, which can improve throughput
    but may slightly increase peak memory usage.

    For NCCL backend, any collectives, even if issued in different streams,
    contend for the same per-device NCCL stream, which is why the relative
    order in which the collectives are issued matters for overlapping. The
    different backward prefetching settings correspond to different orderings.

    - ``BACKWARD_PRE``: This prefetches the next set of parameters before the
      current set of parameter's gradient computation. This improves backward
      pass throughput by overlapping communication (next all-gather) and
      computation (current gradient computation).
    - ``BACKWARD_POST``: This prefetches the next set of parameters after the
      current set of parameter's gradient computation. This may improve
      backward pass throughput by overlapping communication (current
      reduce-scatter) and computation (next gradient computation).
      Specifically, the next all-gather is reordered to be before the current
      reduce-scatter.
    """

    # NOTE: For both modes, the ordering that defines "current" and "next" is
    # not always correct in the current implementation, so this may cause some
    # performance regression for some models.
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


@dataclass
class MixedPrecision:
    """
    This configures FSDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs, and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers. FSDP
            does not shard buffers, casts them to ``buffer_dtype`` in the first
            forward pass, and keeps them in that dtype thereafter. Model
            checkpointing always happens in full precision.
        keep_low_precision_grads (bool): This specifies whether to upcast
            gradients back to the full parameter precision after the backward
            pass. This may be set to ``False`` to save memory if using custom
            optimizers that can perform the optimizer step in ``reduce_dtype``.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision. For buffers, this is only supported for
        ``StateDictType.FULL_STATE_DICT``.

    .. note:: This API is experimental and subject to change.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and FSDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise.
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    keep_low_precision_grads: bool = False


@dataclass
class CPUOffload:
    """
    This configures CPU offloading.

    Attributes:
        offload_params (bool): This specifies whether to offload parameters to
            CPU when not involved in computation. If enabled, this implicitly
            offloads gradients to CPU as well. This is to support the optimizer
            step, which requires parameters and gradients to be on the same
            device.
    """

    offload_params: bool = False
