"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

from dataclasses import dataclass
from enum import auto, Enum

from typing import Optional

import torch

__all__ = [
    "ShardingStrategy",
    "BackwardPrefetch",
    "MixedPrecision",
    "CPUOffload",
    "StateDictType",
    "StateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "OptimStateDictConfig",
    "FullOptimStateDictConfig",
    "LocalOptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "StateDictSettings",
]


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
    - ``HYBRID_SHARD``: Apply ``FULL_SHARD`` within a node, and replicate parameters across
        nodes. This results in reduced communication volume as expensive all-gathers and
        reduce-scatters are only done within a node, which can be more performant for medium
        -sized models.
    - ``_HYBRID_SHARD_ZERO2``: Apply ``SHARD_GRAD_OP`` within a node, and replicate parameters across
        nodes. This is like ``HYBRID_SHARD``, except this may provide even higher throughput
        since the unsharded parameters are not freed after the forward pass, saving the
        all-gathers in the pre-backward.
    """

    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


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

    .. note:: If the increase in peak memory usage from prefetching is an
        issue, you may consider passing ``limit_all_gathers=True`` to the FSDP
        constructor, which may help reduce peak memory usage in some cases.
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
            parameters, inputs (when ``cast_forward_inputs`` or
            ``cast_root_forward_inputs``is set to
            ``True``), and therefore the dtype for computation.
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
            (Default: ``False``)
        cast_forward_inputs (bool): Cast floating point tensors in the forward
            arguments and keyword arguments to ``param_dtype``.
            (Default: ``False``)
        cast_root_forward_inputs (bool): Cast floating point tensors in the forward
            arguments and keyword arguments to ``param_dtype`` for the root FSDP instance.
            It takes precedence over ``cast_forward_inputs`` for the root FSDP instance.
            (Default: ``True``)

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision. For buffers, this is only supported for
        ``StateDictType.FULL_STATE_DICT``.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and FSDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise.

    .. note:: If the user passes a model with ``BatchNorm`` modules and an
        ``auto_wrap_policy`` to the FSDP constructor, then FSDP will disable
        mixed precision for ``BatchNorm`` modules by wrapping them separately
        in their own FSDP instance with mixed precision disabled. This is due
        to some missing low precision ``BatchNorm`` kernels. If the user does
        not use an ``auto_wrap_policy``, then the user must take care to not
        use mixed precision for FSDP instances containing ``BatchNorm``
        modules.

    .. note:: ``MixedPrecision`` has ``cast_root_forward_inputs=True`` and
        ``cast_forward_inputs=False`` by default. For the root FSDP instance,
        its ``cast_root_forward_inputs`` takes precedence over its
        ``cast_forward_inputs``. For non-root FSDP instances, their
        ``cast_root_forward_inputs`` values are ignored. The default setting is
        sufficient for the typical case where each FSDP instance has the same
        ``MixedPrecision`` configuration and only needs to cast inputs to the
        ``param_dtype`` at the beginning of the model's forward pass.

    .. note:: For nested FSDP instances with different ``MixedPrecision``
        configurations, we recommend setting individual ``cast_forward_inputs``
        values to configure casting inputs or not before each instance's
        forward. In such a case, since the casts happen before each FSDP
        instance's forward, a parent FSDP instance should have its non-FSDP
        submodules run before its FSDP submodules to avoid the activation dtype
        being changed due to a different ``MixedPrecision`` configuration.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
            >>> model[1] = FSDP(
            >>>     model[1],
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            >>> )
            >>> model = FSDP(
            >>>     model,
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
            >>> )

        The above shows a working example. On the other hand, if ``model[1]``
        were replaced with ``model[0]``, meaning that the submodule using
        different ``MixedPrecision`` ran its forward first, then ``model[1]``
        would incorrectly see ``float16`` activations instead of ``bfloat16``
        ones.

    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True


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


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT to comply the PyTorch convention.
    ..note::
        FSDP currently supports three types of ``state_dict``:
            1. ``state_dict/load_state_dict`: this pair of APIs return and load
               the non-sharded, unflattened parameters. The semantics is the
               same as using DDP.
            2. ``_local_state_dict/_load_local_state_dict``: this pair of APIs return
               and load local sharded, flattened parameters. The values returned
               by ``_local_state_dict`` can be directly used by FSDP and is only
               meaningful to FSDP (because parameters are flattened). Note that
               these APIs are meant for use via the :func:`state_dict_type`
               context manager as follows:
                   >>> # xdoctest: +SKIP("undefined variables")
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   ...     state = fsdp.state_dict()  # loads local state dict
            3. ``_sharded_state_dict/_load_sharded_state_dict``: this pair of APIs
               return and load sharded, unflattened parameters. The ``state_dict``
               return by ``sharded_state_dict`` can be used by all other parallel
               schemes (resharding may be required).
    """

    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


@dataclass
class StateDictConfig:
    """
    ``StateDictConfig`` is the base class for all state_dict configuration classes.
    Users should instantiate a child version (i.e. ``FullStateDictConfig``) in
    order to configure settings for the particular type of ``state_dict``
    implementation FSDP will use.
    """

    offload_to_cpu: bool = False


@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. Currently, it accepts two parameters,
    ``offload_to_cpu`` and ``rank0_only`` which can be configured to offload
    the full ``state_dict`` to CPU and to materialize the ``state_dict`` on
    rank 0 only. When used, it is recommended to enable both of these flags
    together to optimize memory savings when taking checkpoints. Note that
    this config class is meant for user via the :func:`state_dict_type`
    context manager as follows:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> fsdp = FSDP(model, auto_wrap_policy=...)
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with FullyShardedDataParallel.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>>     # state will be empty on non rank 0 and contain CPU tensors on rank 0.
        >>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
        >>> model = model_fn() # Initialize model on CPU in preparation for wrapping with FSDP
        >>> if dist.get_rank() == 0:
        >>>     # Load checkpoint only on rank 0 to avoid memory redundancy
        >>>     state_dict = torch.load("my_checkpoint.pt")
        >>>     model.load_state_dict(state_dict)
        >>> # All ranks initialize FSDP module as usual. ``sync_module_states`` argument
        >>> # communicates loaded checkpoint states from rank 0 to rest of the world.
        >>> fsdp = FSDP(model, device_id=torch.cuda.current_device(), auto_wrap_policy=..., sync_module_states=True)
        >>> # After this point, all ranks have FSDP model with loaded checkpoint.
    """

    rank0_only: bool = False


@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass


@dataclass
class ShardedStateDictConfig(StateDictConfig):
    pass


@dataclass
class OptimStateDictConfig:
    """
    ``OptimStateDictConfig`` is the base class for all optimizer state_dict
    configuration classes.  Users should instantiate a child version
    (i.e. ``FullOptimStateDictConfig``) in order to configure settings for the
    particular type of ``optim_state_dict`` implementation FSDP will use.
    """

    # TODO: actually use this flag in the _optim_utils.py
    offload_to_cpu: bool = True


@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    rank0_only: bool = False


@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = False


@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    pass


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
