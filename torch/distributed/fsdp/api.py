"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

from dataclasses import dataclass
from enum import auto, Enum

from typing import Optional, Sequence, Type

import torch
from torch.nn.modules.batchnorm import _BatchNorm

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
    This configures explicit backward prefetching, which improves throughput by
    enabling communication and computation overlap in the backward pass at the
    cost of slightly increased memory usage.

    - ``BACKWARD_PRE``: This enables the most overlap but increases memory
      usage the most. This prefetches the next set of parameters *before* the
      current set of parameters' gradient computation. This overlaps the *next
      all-gather* and the *current gradient computation*, and at the peak, it
      holds the current set of parameters, next set of parameters, and current
      set of gradients in memory.
    - ``BACKWARD_POST``: This enables less overlap but requires less memory
      usage. This prefetches the next set of parameters *after* the current
      set of parameters' gradient computation. This overlaps the *current
      reduce-scatter* and the *next gradient computation*, and it frees the
      current set of parameters before allocating memory for the next set of
      parameters, only holding the next set of parameters and current set of
      gradients in memory at the peak.
    - FSDP's ``backward_prefetch`` argument accepts ``None``, which disables
      the backward prefetching altogether. This has no overlap and does not
      increase memory usage. In general, we do not recommend this setting since
      it may degrade throughput significantly.

    For more technical context: For a single process group using NCCL backend,
    any collectives, even if issued from different streams, contend for the
    same per-device NCCL stream, which implies that the relative order in which
    the collectives are issued matters for overlapping. The two backward
    prefetching values correspond to different issue orders.
    """

    # NOTE: For both modes, the ordering that defines "current" and "next" is
    # not always exact in the current implementation. A mistargeted prefetch
    # simply means that the parameter memory is allocated earlier than needed,
    # possibly increasing peak memory usage, but does not affect correctness.
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


@dataclass
class MixedPrecision:
    """
    This configures FSDP-native mixed precision training.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for model
            parameters during forward and backward and thus the dtype for
            forward and backward computation. Outside forward and backward, the
            *sharded* parameters are kept in full precision (e.g. for the
            optimizer step), and for model checkpointing, the parameters are
            always saved in full precision. (Default: ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then this takes on
            the ``param_dtype`` value, still running gradient reduction in low
            precision. This is permitted to differ from ``param_dtype``, e.g.
            to force gradient reduction to run in full precision. (Default:
            ``None``)
        buffer_dtype (Optional[torch.dtype]): This specifies the dtype for
            buffers. FSDP does not shard buffers. Rather, FSDP casts them to
            ``buffer_dtype`` in the first forward pass and keeps them in that
            dtype thereafter. For model checkpointing, the buffers are saved
            in full precision except for ``LOCAL_STATE_DICT``. (Default:
            ``None``)
        keep_low_precision_grads (bool): If ``False``, then FSDP upcasts
            gradients to full precision after the backward pass in preparation
            for the optimizer step. If ``True``, then FSDP keeps the gradients
            in the dtype used for gradient reduction, which can save memory if
            using a custom optimizer that supports running in low precision.
            (Default: ``False``)
        cast_forward_inputs (bool): If ``True``, then this FSDP module casts
            its forward args and kwargs to ``param_dtype``. This is to ensure
            that parameter and input dtypes match for forward computation, as
            required by many ops. This may need to be set to ``True`` when only
            applying mixed precision to some but not all FSDP modules, in which
            case a mixed-precision FSDP submodule needs to recast its inputs.
            (Default: ``False``)
        cast_root_forward_inputs (bool): If ``True``, then the root FSDP module
            casts its forward args and kwargs to ``param_dtype``, overriding
            the value of ``cast_forward_inputs``. For non-root FSDP modules,
            this does not do anything. (Default: ``True``)
        _module_classes_to_ignore: (Sequence[Type[nn.Module]]): This specifies
            module classes to ignore for mixed precision when using an
            ``auto_wrap_policy``: Modules of these classes will have FSDP
            applied to them separately with mixed precision disabled (meaning
            that the final FSDP construction would deviate from the specified
            policy). If ``auto_wrap_policy`` is not specified, then this does
            not do anything. This API is experimental and subject to change.
            (Default: ``(_BatchNorm,)``)

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: Layer norm and batch norm accumulate in ``float32`` even when
        their inputs are in a low precision like ``float16`` or ``bfloat16``.
        Disabling FSDP's mixed precision for those norm modules only means that
        the affine parameters are kept in ``float32``. However, this incurs
        separate all-gathers and reduce-scatters for those norm modules, which
        may be inefficient, so if the workload permits, the user should prefer
        to still apply mixed precision to those modules.

    .. note:: By default, if the user passes a model with any ``_BatchNorm``
        modules and specifies an ``auto_wrap_policy``, then the batch norm
        modules will have FSDP applied to them separately with mixed precision
        disabled. See the ``_module_classes_to_ignore`` argument.

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
    _module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = (_BatchNorm,)


@dataclass
class CPUOffload:
    """
    This configures CPU offloading.

    Attributes:
        offload_params (bool): This specifies whether to offload parameters to
            CPU when not involved in computation. If ``True``, then this
            offloads gradients to CPU as well, meaning that the optimizer step
            runs on CPU.
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
    ``StateDictConfig`` is the base class for all ``state_dict`` configuration
    classes. Users should instantiate a child class (e.g.
    ``FullStateDictConfig``) in order to configure settings for the
    corresponding ``state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict
            values to CPU, and if ``False``, then FSDP keeps them on GPU.
            (Default: ``False``)
    """

    offload_to_cpu: bool = False


@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. We recommend enabling both
    ``offload_to_cpu=True`` and ``rank0_only=True`` when saving full state
    dicts to save GPU memory and CPU memory, respectively. This config class
    is meant to be used via the :func:`state_dict_type` context manager as
    follows:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> fsdp = FSDP(model, auto_wrap_policy=...)
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>>     # `state` will be empty on non rank 0 and contain CPU tensors on rank 0.
        >>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
        >>> model = model_fn() # Initialize model on CPU in preparation for wrapping with FSDP
        >>> if dist.get_rank() == 0:
        >>>     # Load checkpoint only on rank 0 to avoid memory redundancy
        >>>     state_dict = torch.load("my_checkpoint.pt")
        >>>     model.load_state_dict(state_dict)
        >>> # All ranks initialize FSDP module as usual. `sync_module_states` argument
        >>> # communicates loaded checkpoint states from rank 0 to rest of the world.
        >>> fsdp = FSDP(model, device_id=torch.cuda.current_device(), auto_wrap_policy=..., sync_module_states=True)
        >>> # After this point, all ranks have FSDP model with loaded checkpoint.

    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """

    rank0_only: bool = False


@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass


@dataclass
class ShardedStateDictConfig(StateDictConfig):
    """
    ``ShardedStateDictConfig`` is a config class meant to be used with
    ``StateDictType.SHARDED_STATE_DICT``.

    Attributes:
        _use_dtensor (bool): If ``True``, then FSDP saves the state dict values
            as ``DTensor``, and if ``False``, then FSDP saves them as
            ``ShardedTensor``. (Default: ``False``)

    .. warning:: ``_use_dtensor`` is a private field of :class:`ShardedStateDictConfig`
      and it is used by FSDP to determine the type of state dict values. Users should not
      manually modify ``_use_dtensor``.
    """

    _use_dtensor: bool = False


@dataclass
class OptimStateDictConfig:
    """
    ``OptimStateDictConfig`` is the base class for all ``optim_state_dict``
    configuration classes.  Users should instantiate a child class (e.g.
    ``FullOptimStateDictConfig``) in order to configure settings for the
    corresponding ``optim_state_dict`` type supported by FSDP.

    Attributes:
        offload_to_cpu (bool): If ``True``, then FSDP offloads the state dict's
            tensor values to CPU, and if ``False``, then FSDP keeps them on the
            original device (which is GPU unless parameter CPU offloading is
            enabled). (Default: ``True``)
    """

    # TODO: actually use this flag in the _optim_utils.py
    offload_to_cpu: bool = True


@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    """
    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """

    rank0_only: bool = False


@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = False


@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    """
    ``ShardedOptimStateDictConfig`` is a config class meant to be used with
    ``StateDictType.SHARDED_STATE_DICT``.

    Attributes:
        _use_dtensor (bool): If ``True``, then FSDP saves the state dict values
            as ``DTensor``, and if ``False``, then FSDP saves them as
            ``ShardedTensor``. (Default: ``False``)

    .. warning:: ``_use_dtensor`` is a private field of :class:`ShardedOptimStateDictConfig`
      and it is used by FSDP to determine the type of state dict values. Users should not
      manually modify ``_use_dtensor``.
    """

    _use_dtensor: bool = False


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
