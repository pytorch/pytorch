import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _get_param_to_fqns,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    HandleTrainingState,
    TrainingState,
)
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _get_default_comm_hook,
    _init_buffer_state,
    _init_core_state,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
)
from torch.distributed.fsdp._runtime_utils import (
    _cast_buffers_to_dtype_and_device,
    _clear_grads_if_needed,
    _get_buffers_and_dtypes_for_checkpoint,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _reshard,
    _reshard_grads,
    _root_pre_forward,
    _should_free_in_backward,
    _unshard,
    _unshard_grads,
    _wait_for_computation_stream,
)
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

from ._optim_utils import (
    _broadcast_pos_dim_tensor_states,
    _broadcast_processed_optim_state_dict,
    _flatten_optim_state_dict,
    _get_param_id_to_param,
    _get_param_id_to_param_from_optim_input,
    _get_param_to_param_id,
    _get_param_to_param_id_from_optim_input,
    _optim_state_dict,
    _process_pos_dim_tensor_state,
    _rekey_sharded_optim_state_dict,
)
from ._state_dict_utils import (
    _post_load_state_dict_hook,
    _post_state_dict_hook,
    _pre_load_state_dict_hook,
)
from ._utils import p_assert
from .flat_param import FlatParameter, FlatParamHandle
from .wrap import ParamExecOrderWrapPolicy


_TORCH_FX_AVAIL = True
if not hasattr(torch, "fx"):
    _TORCH_FX_AVAIL = False
if _TORCH_FX_AVAIL:
    from ._symbolic_trace import _init_execution_info, _patch_tracer, TracingConfig


__all__ = [
    "FullyShardedDataParallel",
    "StateDictType",
    "StateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "OptimStateKeyType",
]


FLAT_PARAM = "_flat_param"


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


class OptimStateKeyType(Enum):
    PARAM_NAME = auto()
    PARAM_ID = auto()


class FullyShardedDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters across data parallel workers. This
    is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    .. warning::
        The optimizer must be initialized *after* the module has been wrapped,
        since FSDP will shard parameters in-place and this will break any
        previously initialized optimizers.

    .. warning::
        If the destination CUDA device has ID ``dev_id``, either (1)
        ``module`` should already be placed on that device, (2) the device
        should be set using ``torch.cuda.set_device(dev_id)``, or (3)
        ``dev_id`` should be passed into the ``device_id`` constructor
        argument. This FSDP instance's compute device will be that destination
        device. For (1) and (3), the FSDP initialization always occurs on GPU.
        For (2), the FSDP initialization happens on ``module`` 's current
        device, which may be CPU.

    .. warning::
        FSDP currently does not support gradient accumulation outside
        ``no_sync()`` when using CPU offloading. Trying to do so yields
        incorrect results since FSDP will use the newly-reduced gradient
        instead of accumulating with any existing gradient.

    .. warning::
        Changing the original parameter variable names after construction will
        lead to undefined behavior.

    .. warning::
        Passing in `sync_module_states=True` flag requires module to be put
        on GPU, or to use ``device_id`` argument to specify a CUDA device that
        FSDP will move module to. This is because ``sync_module_states=True``
        requires GPU communication.

    .. warning::
        As of PyTorch 1.12, FSDP only offers limited support for shared parameters
        (for example, setting one ``Linear`` layer's weight to another's). In
        particular, modules that share parameters must be wrapped as part of the
        same FSDP unit. If enhanced shared parameter support is needed for your
        use case, please ping https://github.com/pytorch/pytorch/issues/77724

    .. note:
        Attempting to run the forward pass of a submodule that is contained in an
        FSDP instance is not supported and will result in errors. This is because the
        submodule's parameters will be sharded, but it itself is not an FSDP instance,
        so its forward pass will not all-gather the full parameters appropriately.
        This could potentially happen when attempting to run only the encoder of a
        encoder-decoder model, and the encoder is not wrapped in its own FSDP instance. To
        resolve this, please wrap the submodule in its own FSDP unit.

    .. note::
        Inputs into FSDP ``forward`` function will be moved to compute device
        (same device FSDP module is on) before running ``forward``, so user does
        not have to manually move inputs from CPU -> GPU.

    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[ProcessGroup]):
            This is the process group used for collective communications.
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy used by FSDP, which may trade
            off memory saving and communication overhead. See
            :class:`ShardingStrategy` for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]):
            A callable specifying a policy to recursively wrap layers with FSDP.
            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``size_based_auto_wrap_policy`` written in ``torch.distributed.fsdp.wrap`` is
            an example of ``auto_wrap_policy`` callable, this policy wraps layers
            with the number of parameters larger than 100M. ``transformer_auto_wrap_policy``
            written in ``torch.distributed.fsdp.wrap`` is an example of ``auto_wrap_policy``
            callable for transformer-like model architectures. Users can supply the customized
            ``auto_wrap_policy`` callable that should accept following arguments:
            ``module: nn.Module``, ``recurse: bool``, ``unwrapped_params: int``, and return
            a ``bool`` specifying whether the passed in ``module``` should be wrapped
            (if ``recurse=False``) or whether we should recurse down the subgraph of ``module``
            children (if ``recurse=True``). Extra customized arguments could be added to
            the customized ``auto_wrap_policy`` callable as well. It is a good practice to
            print out the sharded model and check whether the sharded model is what
            the application wants and then adjust accordingly.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     unwrapped_params: int,
                >>>     # These are customizable for this policy function.
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return unwrapped_params >= min_num_params
                >>> # Configure a custom min_num_params
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=1e5)

        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. See
            :class:`BackwardPrefetch` for details. (Default: ``BACKWARD_PRE``)
        mixed_precision (Optional[MixedPrecision]):
            This configures native mixed precision for FSDP. If this is set to
            ``None``, then no mixed precision is used. Otherwise, parameter,
            buffer, and gradient reduction dtypes can be set. See
            :class:`MixedPrecision` for details. (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should be initialized
            onto an actual device. Note that as of v1.12, we detect modules on the meta
            device via ``is_meta`` check and apply a default initialization that calls
            ``reset_parameters`` method on the passed in ``nn.Module`` if ``param_init_fn``
            is not specified, otherwise we run ``param_init_fn`` to initialize the passed
            in ``nn.Module``. In particular, this means that if ``is_meta=True`` for any
            module parameters for modules that will be wrapped with FSDP and ``param_init_fn``
            is not specified, we assume your module properly implements a ``reset_parameters()``
            and will throw errors if not. Note that additionally, we offer support for modules
            initialized with torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init`` API. In this case, deferred modules would be initialized
            by a default initialization function that calls torchdistX's
            ``materialize_module``, or the passed in ``param_init_fn``, if it is not
            ``None``. The same ``Callable`` is applied to initialize all meta modules.
            Note that this initialization function is applied before doing any FSDP sharding
            logic.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module):
                >>>     # responsible for initializing a module, such as with reset_parameters
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or ``torch.device``
            describing the CUDA device the FSDP module should be moved to determining where
            initialization such as sharding takes place. If this argument is not specified
            and ``module`` is on CPU, we issue a warning mentioning that this argument can
            be specified for faster initialization. If specified, resulting FSDP instances
            will reside on this device, including moving ignored modules' parameters if
            needed. Note that if ``device_id`` is specified but ``module`` is already on a
            different CUDA device, an error will be thrown. (Default: ``None``)
        sync_module_states (bool): If ``True``, each individually wrapped FSDP unit will broadcast
            module parameters from rank 0 to ensure they are the same across all ranks after
            initialization. This helps ensure model parameters are the same across ranks
            before starting training, but adds communication overhead to ``__init__``, as at least
            one broadcast is triggered per individually wrapped FSDP unit.
            This can also help load checkpoints taken by ``state_dict`` and to be loaded by
            ``load_state_dict`` in a memory efficient way. See documentation for
            :class:`FullStateDictConfig` for an example of this. (Default: ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next upcoming all-gather while executing in the forward pass.
            This may improve communication and computation overlap for CPU
            bound workloads. This should only be used for static graph models
            since the forward order is fixed based on the first iteration's
            execution. (Default: ``False``)
        limit_all_gathers (bool): If ``False``, then FSDP allows the CPU
            thread to schedule all-gathers without any extra synchronization.
            If ``True``, then FSDP explicitly synchronizes the CPU thread to
            prevent too many in-flight all-gathers. This ``bool`` only affects
            the sharded strategies that schedule all-gathers. Enabling this can
            help lower the number of CUDA malloc retries.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Callable] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = False,
        use_orig_params: bool = False,
    ):
        if isinstance(auto_wrap_policy, ParamExecOrderWrapPolicy):
            self._init_param_exec_order_wrap_policy(
                module=module,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                ignored_modules=ignored_modules,
                param_init_fn=param_init_fn,
                device_id=device_id,
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
                use_orig_params=use_orig_params,
            )
            return

        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()

        _init_ignored_module_states(self, module, ignored_modules)
        if auto_wrap_policy is not None:
            auto_wrap_kwargs = {
                "module": module,
                "auto_wrap_policy": auto_wrap_policy,
                "wrapper_cls": FullyShardedDataParallel,
                "ignored_modules": self._ignored_modules,
                "ignored_params": self._ignored_params,
                "only_wrap_children": True,  # avoid double wrapping the root
            }
            fsdp_kwargs = {
                "process_group": process_group,
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "mixed_precision": mixed_precision,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "sync_module_states": sync_module_states,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
                "use_orig_params": use_orig_params,
            }
            _auto_wrap(auto_wrap_kwargs, fsdp_kwargs, FullyShardedDataParallel)

        _init_process_group_state(self, process_group)
        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            sharding_strategy,
            mixed_precision,
            cpu_offload,
            limit_all_gathers,
            use_orig_params,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        _init_runtime_state(self)
        _init_prefetching_state(self, backward_prefetch, forward_prefetch)
        _init_buffer_state(self, module)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
            sync_module_states,
            FullyShardedDataParallel,
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            self._register_flat_param()

        # Delete to avoid keeping references after the constructor
        delattr(self, "_ignored_params")

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)  # TODO: currently a no-op; need to refactor below
        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config = FullStateDictConfig()
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            _pre_load_state_dict_hook, with_module=True
        )
        self.register_load_state_dict_post_hook(_post_load_state_dict_hook)

    @property
    def module(self) -> nn.Module:
        """
        Returns the wrapped module (like :class:`DistributedDataParallel`).
        """
        # FSDP's `.module` must refer to the innermost wrapped module when
        # composing with other module wrappers in order for state dict to work
        if isinstance(self._fsdp_wrapped_module, ActivationWrapper):
            return getattr(self._fsdp_wrapped_module, _CHECKPOINT_WRAPPED_MODULE)
        return self._fsdp_wrapped_module

    @property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
        return hasattr(self, "_handles") and len(self._handles) > 0

    @property
    def _flat_param(self) -> Optional[FlatParameter]:
        return self._handles[0].flat_param if self._handles else None

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        if hasattr(self, FSDP_WRAPPED_MODULE):
            return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]
        return super().__getitem__(key)

    def check_is_root(self) -> bool:
        _lazy_init(self, self)
        assert self._is_root is not None
        return self._is_root

    @staticmethod
    def fsdp_modules(
        module: nn.Module,
        root_only: bool = False,
    ) -> List["FullyShardedDataParallel"]:
        """
        Returns all nested FSDP instances, possibly including ``module`` itself
        and only including FSDP root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
        return [
            submodule
            for submodule in module.modules()
            if isinstance(submodule, FullyShardedDataParallel)
            and (not root_only or submodule.check_is_root())
        ]

    @staticmethod
    def _fsdp_handles(module: nn.Module) -> List[FlatParamHandle]:
        """
        Returns all nested FSDP instances' handles in the module hierarchy
        rooted at ``module``.
        """
        return [
            handle
            for fsdp_module in FullyShardedDataParallel.fsdp_modules(module)
            for handle in fsdp_module._handles
        ]

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState.IDLE)
        with self._summon_full_params(recurse=False, writeback=True):
            ret = super().apply(fn)

        # Reset lazy init that might be called by _summon_full_params, since
        # it could have set is_root incorrectly for non-root FSDP instances.
        if uninitialized and self._is_root:
            for module in self.fsdp_modules(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_params(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        parameters or not.
        """
        return self.mixed_precision.param_dtype is not None

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        buffers or not.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _mixed_precision_enabled_for_reduce(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        gradient reduction or not.
        """
        return self.mixed_precision.reduce_dtype is not None

    def _mixed_precision_keep_low_precision_grads(self) -> bool:
        return (
            self.mixed_precision is not None
            and self.mixed_precision.keep_low_precision_grads
        )

    def _low_precision_hook_enabled(self) -> bool:
        """
        Wether a low precision hook is registered or not.
        """
        return (
            self._communication_hook is not None
            and self._communication_hook in LOW_PRECISION_HOOKS
        )

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        """
        self._is_root: Optional[bool] = None

    @staticmethod
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
    ) -> Tuple[StateDictType, StateDictConfig]:
        """
        Set the ``state_dict_type`` and the corresponding (optional)
        configurations of all the descendant FSDP modules of the target module.
        The target module does not have to be a FSDP module. If the target
        module is a FSDP module, its ``state_dict_type`` will also be changed.

        .. note:: This API should be called for only the top-level (root)
            module.

        .. note:: This API enables users to transparently use the conventional
            ``state_dict`` API to take model checkpoints in cases where the
            root FSDP module is wrapped by another ``nn.Module``. For example,
            the following will ensure ``state_dict`` is called on all non-FSDP
            instances, while dispatching into `sharded_state_dict` implementation
            for FSDP:

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>>     ShardedStateDictConfig(offload_to_cpu=True),
            >>> )
            >>> checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
        """
        _state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }

        prev_state_dict_type = None
        prev_state_dict_config = None
        # Use the default config if a state_dict config is not set.
        if state_dict_config is None:
            state_dict_config = _state_dict_type_to_config[state_dict_type]()
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = submodule._state_dict_type
            if prev_state_dict_config is None:
                prev_state_dict_config = submodule._state_dict_config
            if prev_state_dict_type != submodule._state_dict_type:
                raise RuntimeError("All FSDP module should the same state_dict_type.")
            if not isinstance(
                submodule._state_dict_config, type(prev_state_dict_config)
            ):
                raise RuntimeError(
                    "All FSDP modules should have the same type of state_dict_config."
                )

            expected_state_dict_config_type = _state_dict_type_to_config[
                state_dict_type
            ]
            if expected_state_dict_config_type != type(state_dict_config):
                raise RuntimeError(
                    f"Expected state_dict_config of type {expected_state_dict_config_type} "
                    f"but got {type(state_dict_config)}"
                )
            submodule._state_dict_type = state_dict_type
            submodule._state_dict_config = state_dict_config

        return prev_state_dict_type, prev_state_dict_config

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
    ) -> Generator:
        """
        A context manager to set the ``state_dict_type`` of all the descendant
        FSDP modules of the target module. This context manager has the same
        functions as :meth:`set_state_dict_type`. Read the document of
        :meth:`set_state_dict_type` for the detail.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>> ):
            >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
        """
        prev_state_dict_type = None
        prev_state_dict_config = None
        try:
            (
                prev_state_dict_type,
                prev_state_dict_config,
            ) = FullyShardedDataParallel.set_state_dict_type(
                module, state_dict_type, state_dict_config
            )
            yield
        except Exception as e:
            raise e
        else:
            assert prev_state_dict_type is not None
            assert prev_state_dict_config is not None
        finally:
            if prev_state_dict_type is not None and prev_state_dict_config is not None:
                FullyShardedDataParallel.set_state_dict_type(
                    module, prev_state_dict_type, prev_state_dict_config
                )

    def _convert_to_wrapped_module_name(self, module_name: str) -> str:
        module_name = module_name.replace(f"{FSDP_PREFIX}", "")
        module_name = module_name.replace(f"{FSDP_WRAPPED_MODULE}", "")
        if module_name:
            module_name = f"{module_name}."
        # Activation checkpoint adds a prefix that has to be
        # removed as well.
        module_name = module_name.replace(_CHECKPOINT_PREFIX, "")
        return module_name

    @property
    def _param_fqns(self) -> Iterator[Tuple[str, str, str]]:
        if not self._has_params:
            return
        for param_name, module_name in self._handles[0].parameter_module_names():
            module_name = self._convert_to_wrapped_module_name(module_name)
            fqn = f"{module_name}{param_name}"
            yield fqn, param_name, module_name

    @property
    def _shared_param_fqns(self) -> Iterator[Tuple[str, str, str]]:
        for param_name, module_name in self._handles[0].shared_parameter_module_names():
            module_name = self._convert_to_wrapped_module_name(module_name)
            fqn = f"{module_name}{param_name}"
            yield fqn, param_name, module_name

    def state_dict(self, *args, **kwargs):
        """
        This is the entry point of all three FSDP ``state_dict`` APIs: full,
        local, and sharded. For the full state dict
        (``StateDictType.FULL_STATE_DICT``), FSDP attempts to unshard the model
        on all ranks, which may result in an OOM error if the full model cannot
        fit on a single GPU. In that case, users may pass in a
        :class:`FullStateDictConfig` to only save the checkpoint on rank 0 and/
        or to offload it to CPU memory layer by layer, enabling much larger
        checkpoints. If the full model cannot fit in CPU memory, then users may
        instead take a local state dict (``StateDictType.LOCAL_STATE_DICT``)
        that only saves the local shard of the model. The sharded state dict
        (``StateDictType.SHARDED_STATE_DICT``) saves the model parameters as
        ``ShardedTensor`` s. The ``state_dict`` type can be configured using
        the :meth:`state_dict_type` context manager.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> torch.cuda.set_device(device_id)
            >>> my_module = nn.Linear(...)
            >>> sharded_module = FSDP(my_module)
            >>> full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            >>>     full_dict = sharded_module.state_dict()
            >>> full_dict.keys()
            >>> odict_keys(['weight', 'bias'])
            >>> # using local state dict
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.LOCAL_STATE_DICT):
            >>>     local_dict = sharded_module.state_dict()
            >>> local_dict.keys()
            >>> odict_keys(['flat_param', 'inner.flat_param'])

        .. warning:: This needs to be called on all ranks since it uses
            collective communications.
        """
        # TODO (rohan-varma): separate these out once a state_dict pre-hook
        # is available.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _lazy_init(self, self)
        if self._is_root:
            _clear_grads_if_needed(self._fsdp_handles(self))
        if self._state_dict_type == StateDictType.FULL_STATE_DICT:
            # Get config args
            full_state_dict_config = (
                self._state_dict_config
                if self._state_dict_config is not None
                else FullStateDictConfig()
            )
            rank0_only = full_state_dict_config.rank0_only
            offload_to_cpu = full_state_dict_config.offload_to_cpu
            summon_ctx = (
                self._summon_full_params(
                    recurse=False,
                    writeback=False,
                    offload_to_cpu=offload_to_cpu,
                    rank0_only=rank0_only,
                )
                if self.training_state != TrainingState.SUMMON_FULL_PARAMS
                else contextlib.suppress()
            )
            with summon_ctx:
                # Since buffers stay in their low precision throughout runtime,
                # we must explicitly restore them to their original dtypes for
                # model checkpointing. We have the root module cast for all
                # submodules.
                # TODO: Investigate if this can and should be refactored into
                # `summon_full_params()`.
                if self._is_root and self._mixed_precision_enabled_for_buffers():
                    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_checkpoint(
                        self, self
                    )
                    _cast_buffers_to_dtype_and_device(
                        buffers, buffer_dtypes, self.compute_device
                    )
                state_dict = super().state_dict(*args, **kwargs)

            # TODO: support offload to CPU in post state dict hook.
            if not rank0_only or self.rank == 0:
                return state_dict
            else:
                return {}

        elif (
            self._state_dict_type == StateDictType.LOCAL_STATE_DICT
            or self._state_dict_type == StateDictType.SHARDED_STATE_DICT
        ):
            if self._has_params and not self._handles[0].uses_sharded_strategy:
                raise RuntimeError(
                    "sharded_state_dict/local_state_dict can only be called "
                    "when parameters are flatten and sharded."
                )
            return super().state_dict(*args, **kwargs)
        else:
            raise ValueError(f"Unknown StateDictType {self._state_dict_type}.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the forward pass for the wrapped module, inserting FSDP-specific
        pre- and post-forward sharding logic.
        """
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, *args, **kwargs)
            unused = None
            unshard_fn = functools.partial(_pre_forward_unshard, self, self._handles)
            reshard_fn = functools.partial(_post_forward_reshard, self, self._handles)
            _pre_forward(
                self, self._handles, unshard_fn, self._fsdp_wrapped_module, unused
            )
            for handle in self._handles:
                p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, self._handles, reshard_fn, unused, unused, output
            )

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
        with_grads: bool = False,
    ) -> Generator:
        r"""A context manager to expose full params for FSDP instances.
        Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking. It can take a non-FSDP
        module and will summon full params for all contained FSDP modules as
        well as their children, depending on the ``recurse`` argument.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when ``world_size == 1``, or ``NO_SHARD``
            config, the modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        .. warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
            result in full parameters being redundantly copied to CPU memory for
            GPUs that reside on the same machine, which may incur the risk of
            CPU OOM. It is recommended to use ``offload_to_cpu`` with
            ``rank0_only=True``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True).
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exits;
                disabling this can be slightly more efficient (default: True)
            rank0_only (bool, Optional): if ``True``, full parameters are
                materialized on only global rank 0. This means that within the
                context, only rank 0 will have full parameters and the other
                ranks will have sharded parameters. Note that setting
                ``rank0_only=True`` with ``writeback=True`` is not supported,
                as model parameter shapes will be different across ranks
                within the context, and writing to them can lead to
                inconsistency across ranks when the context is exited.
            offload_to_cpu (bool, Optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1 or ``NO_SHARD`` config). It is recommended
                to use ``offload_to_cpu`` with ``rank0_only=True`` to avoid
                redundant copies of model parameters being offloaded to the same CPU memory.
            with_grads (bool, Optional): If ``True``, gradients are also
                unsharded with the parameters. Currently, this is only
                supported when passing ``use_orig_params=True`` to the FSDP
                constructor and ``offload_to_cpu=False`` to this method.
                (Default: ``False``)
        """
        # Note that we specify root_only as FSDP roots will handle summoning
        # child FSDP instances based on recurse argument.
        root_fsdp_modules = FullyShardedDataParallel.fsdp_modules(
            module, root_only=True
        )
        # Summon all params for all FSDP instances
        with contextlib.ExitStack() as stack:
            for module in root_fsdp_modules:
                stack.enter_context(
                    module._summon_full_params(
                        recurse=recurse,
                        writeback=writeback,
                        rank0_only=rank0_only,
                        offload_to_cpu=offload_to_cpu,
                        with_grads=with_grads,
                    )
                )
            # Yield to the caller, with full params in all FSDP instances.
            yield
        # Exiting from the ExitStack will reshard all params.
        return

    @contextlib.contextmanager
    def _summon_full_params(
        self,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
        with_grads: bool = False,
    ):
        if with_grads and (offload_to_cpu or not self._use_orig_params):
            raise NotImplementedError(
                f"with_grads={with_grads} "
                f"use_orig_params={self._use_orig_params} "
                f"offload_to_cpu={offload_to_cpu} "
                f"is not supported yet"
            )
        if writeback and rank0_only:
            raise ValueError(
                "writeback=True and rank0_only=True is not supported, as model "
                "parameter shapes will be different across ranks, and writing "
                "to them can lead to inconsistencies across ranks when the "
                "context is exited."
            )
        if offload_to_cpu and not rank0_only:
            warnings.warn(
                "offload_to_cpu and rank0_only=False will result in "
                "full parameters being redundantly copied to CPU memory for "
                "GPUs that reside on the same machine, which may incur the risk of "
                "CPU OOM. It is recommended to use ``offload_to_cpu`` with "
                "rank0_only=True."
            )

        if recurse:
            with contextlib.ExitStack() as stack:
                for module in self.fsdp_modules(self):
                    stack.enter_context(
                        module._summon_full_params(
                            recurse=False,
                            writeback=writeback,
                            rank0_only=rank0_only,
                            offload_to_cpu=offload_to_cpu,
                            with_grads=with_grads,
                        )
                    )
                yield
            return

        torch.cuda.synchronize()
        _lazy_init(self, self)
        self._assert_state([TrainingState.IDLE])
        for handle in self._handles:
            assert handle._training_state == HandleTrainingState.IDLE
        self.training_state = TrainingState.SUMMON_FULL_PARAMS
        for handle in self._handles:
            handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS

        if self._is_root:
            _clear_grads_if_needed(self._fsdp_handles(self))
        free_unsharded_flat_params = [
            handle.needs_unshard() for handle in self._handles
        ]
        # No need to call `wait_stream()` since we unshard in the computation
        # stream directly
        computation_stream = torch.cuda.current_stream()
        _unshard(self, self._handles, computation_stream, computation_stream)
        if with_grads:
            _unshard_grads(self._handles)

        if rank0_only and self.rank != 0:
            # Free the unsharded flattened parameter early
            _reshard(self, self._handles, free_unsharded_flat_params)
            if with_grads:
                _reshard_grads(self._handles)
            try:
                yield
            finally:
                self.training_state = TrainingState.IDLE
                for handle in self._handles:
                    handle._training_state = HandleTrainingState.IDLE
        else:
            # Unflatten the unsharded flattened parameters
            with contextlib.ExitStack() as stack:
                # Invariant: rank == 0 or !rank0_only
                for handle in self._handles:
                    if offload_to_cpu and handle.uses_sharded_strategy:
                        stack.enter_context(handle.to_cpu())
                        # TODO (awgu): Since PyTorch enforces that a parameter
                        # and its gradients need to match metadata (e.g.
                        # device), we must move gradients to CPU *after* we
                        # move parameters.
                # TODO (awgu): This FPW call assumes 1 `FlatParameter`
                if not self._use_orig_params:
                    stack.enter_context(self._unflatten_as_params())
                try:
                    yield
                finally:
                    stack.close()
                    if writeback:
                        self._writeback_to_local_shard(self._handles, with_grads)
                    _reshard(self, self._handles, free_unsharded_flat_params)
                    if with_grads:
                        _reshard_grads(self._handles)
                    self.training_state = TrainingState.IDLE
                    for handle in self._handles:
                        handle._training_state = HandleTrainingState.IDLE

    @torch.no_grad()
    def _writeback_to_local_shard(
        self,
        handles: List[FlatParamHandle],
        writeback_grad: bool,
    ):
        """
        For each handle, writes back the this rank's shard of the unsharded
        flattened parameter to the sharded flattened parameter. If
        ``writeback_grad=True``, then writes back to the sharded gradient as
        well.

        Precondition: Each handle's ``FlatParameter`` 's data points to the
        padded unsharded flattened parameter.
        """
        for handle in handles:
            # For `NO_SHARD`, `_local_shard` is the unsharded flattened
            # parameter and `grad` is the unsharded gradient, so there is no
            # need to writeback for either
            if not handle.uses_sharded_strategy:
                continue
            assert (
                handle.flat_param.ndim == 1
            ), f"Expects `flat_param` to be flattened but got {handle.flat_param.shape}"

            # Get the unpadded shard instead of the padded shard to persist
            # user changes to the padding (though FSDP does not explicitly
            # support this)
            param_shard, _ = FlatParamHandle._get_unpadded_shard(
                handle.flat_param,
                handle.rank,
                handle.world_size,
            )
            handle.flat_param._local_shard[: param_shard.numel()].copy_(param_shard)
            if writeback_grad:
                existing_grad = handle.sharded_grad
                if existing_grad is not None:
                    grad_shard, _ = FlatParamHandle._get_unpadded_shard(
                        handle.flat_param.grad,
                        handle.rank,
                        handle.world_size,
                    )
                    existing_grad[: grad_shard.numel()].copy_(grad_shard)

    @contextlib.contextmanager
    def _unflatten_as_params(self) -> Generator:
        """
        Assumes that the flattened parameter is unsharded. When in the context,
        de-registers the flattened parameter and unflattens the original
        parameters as ``nn.Parameter`` views into the flattened parameter.
        After the context, re-registers the flattened parameter and restores
        the original parameters as ``Tensor`` views into the flattened
        parameter.
        """
        if not self._handles:
            yield
        else:
            self._deregister_flat_param()
            try:
                with self._handles[0].unflatten_as_params():
                    yield
            finally:
                if not self._handles[0]._use_orig_params:
                    self._register_flat_param()

    def _register_flat_param(self):
        """
        Registers the flattened parameter to the wrapped module, making it
        visible to ``nn.Module`` methods.

        We do not use :meth:`nn.Module.register_parameter` because we want
        ``FLAT_PARAM`` to always be an attribute but dynamically change whether
        it is visible to ``nn.Module`` methods.
        """
        if self._has_params:
            self.module._parameters[FLAT_PARAM] = self._handles[0].flat_param

    def _deregister_flat_param(self):
        """
        De-registers the flattened parameter from the wrapped module, hiding it
        from ``nn.Module`` methods.

        We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
        attribute but dynamically change whether it is visible to ``nn.Module``
        methods.
        """
        if self._has_params:
            self.module._parameters.pop(FLAT_PARAM, None)

    @contextlib.contextmanager
    def _deregister_orig_params_ctx(self):
        """
        This deregisters the original parameters and exposes the
        :class:`FlatParameter` s. If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method shouuld
        only be called when using the original parameters.
        """
        p_assert(
            self._use_orig_params,
            "`_deregister_orig_params_ctx()` should only be called when "
            "`_use_orig_params=True`",
        )
        for fsdp_module in self.fsdp_modules(self):
            fsdp_module._deregister_orig_params()
        try:
            yield
        finally:
            for fsdp_module in self.fsdp_modules(self):
                fsdp_module._register_orig_params()

    def _deregister_orig_params(self):
        """
        Deregisters the original parameters; registers the ``FlatParameter``.
        """
        p_assert(
            len(self._handles) <= 1,
            "Expects <=1 handle per FSDP instance; needs to be refactored "
            "for >1 handle (e.g. non-recursive wrapping)",
        )
        if not self._handles:
            return
        handle = self._handles[0]
        p_assert(
            handle._use_orig_params,
            f"Inconsistent `_use_orig_params` -- FSDP: {self._use_orig_params} "
            f"handle: {handle._use_orig_params}",
        )
        handle._deregister_orig_params()
        self._register_flat_param()

    def _register_orig_params(self):
        """
        Deregisters the ``FlatParameter``; registers the original parameters.
        """
        if not self._handles:
            return
        handle = self._handles[0]
        self._deregister_flat_param()
        if handle.is_sharded(handle.flat_param):
            handle._use_sharded_views()
            handle._use_sharded_grad_views()
        else:
            handle._use_unsharded_views(as_params=True)

    def _apply(self, *args, **kwargs):
        """
        When using the original parameters, this deregisters the original
        parameters and exposes the :class:`FlatParameter` s before calling
        ``_apply()``.
        """
        # When using the original parameters: Since (1) the `FlatParameter`s
        # own the storage and (2) `_apply()` is the subroutine underlying the
        # most common storage-changing ops like `to()` and `cuda()`, we
        # override `_apply()` to have the storage change directly performed on
        # the `FlatParameter`s instead of applying to the original parameters
        # and then writing back to the `FlatParameter`s.
        with (
            self._deregister_orig_params_ctx()
            if self._use_orig_params
            else contextlib.suppress()
        ):
            return super()._apply(*args, **kwargs)

    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Overrides :meth:`named_buffers()` to intercept buffer names and
        remove all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = (
            self.training_state == TrainingState.SUMMON_FULL_PARAMS
            or self._use_orig_params
        )
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                buffer_name = buffer_name.replace(FSDP_PREFIX, "")
            yield (buffer_name, buffer)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = (
            self.training_state == TrainingState.SUMMON_FULL_PARAMS
            or self._use_orig_params
        )
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        assert self._is_root, "_wait_for_post_backward can only be called on root."
        # Root's training state might be backward_pre or backward_post depending on
        # if root parameter's post backward hook was called. The post-backward hook
        # may not have been called if gradient was not computed for this param/FSDP
        # module.

        if self._sync_gradients:
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.cpu_offload.offload_params:
                # We need to wait for the non-blocking GPU ->
                # CPU grad transfers to finish. We need to do this for GPU -> CPU
                # copies because when grad is on CPU, it won't wait for any CUDA
                # stream to finish GPU -> CPU copies unless we explicitly block the
                # host-side with synchronize().
                torch.cuda.current_stream().synchronize()
        self._exec_order_data.next_iter()

        # A backward pass is done, clean up below.
        def _catch_all_reshard(fsdp_module: FullyShardedDataParallel) -> None:
            """
            Reshards full parameters that may have not been resharded in
            post_backward_hook. This can happen when an FSDP module's output
            is used in forward so its pre-backward fires unsharding the param,
            but post-backward does not fire since the output was not ultimately
            used in loss computation so FSDP parameter did not get a gradient.
            """
            # Note that we wrap resharding logic in a try-catch as a defensive
            # approach, as if an error is thrown, we are in the backwards pass,
            # and autograd would not print out much useful info about the actual
            # error hit.
            try:
                free_unsharded_flat_params: List[bool] = []
                handles_to_reshard: List[FlatParamHandle] = []
                for handle in fsdp_module._handles:
                    # TODO: This already-resharded check is brittle:
                    # https://github.com/pytorch/pytorch/issues/83956
                    already_resharded = (
                        handle.flat_param.data_ptr()
                        == handle.flat_param._local_shard.data_ptr()
                    )
                    if already_resharded:
                        continue
                    free_unsharded_flat_params.append(
                        _should_free_in_backward(fsdp_module, handle)
                    )
                    handles_to_reshard.append(handle)
                _reshard(self, handles_to_reshard, free_unsharded_flat_params)
            except Exception as e:
                p_assert(
                    False,
                    f"Got exception while resharding module {fsdp_module}: {str(e)}",
                    raise_assertion_error=False,
                )
                raise e

        def _finalize_params(fsdp_module: FullyShardedDataParallel) -> None:
            """Helper used below on all fsdp modules."""
            for handle in fsdp_module._handles:
                p = handle.flat_param
                if p.requires_grad:
                    if hasattr(p, "_post_backward_hook_state"):
                        p_assert(
                            len(p._post_backward_hook_state) == 2,  # type: ignore[attr-defined]
                            "p._post_backward_hook_state fields are not valid.",
                        )
                        p._post_backward_hook_state[1].remove()  # type: ignore[attr-defined]
                        delattr(p, "_post_backward_hook_state")
                    if not self._sync_gradients:
                        # Preserve the gradient accumulation state if not
                        # synchronizing gradients: `p.grad` remains the
                        # unsharded gradient accumulated from prior `no_sync()`
                        # iterations, and `p._saved_grad_shard` remains the
                        # sharded gradient from the last synchronized iteration
                        continue
                    handle.prepare_gradient_for_optim()
                    p_assert(
                        hasattr(p, "_post_backward_called"),
                        "Expected flag _post_backward_called to be set on param.",
                    )
                    # Reset _post_backward_called in preparation for the next iteration.
                    p._post_backward_called = False

        # Update root and nested FSDP's hooks and flags.
        for m in self.fsdp_modules(self):  # includes self
            _catch_all_reshard(m)
            _finalize_params(m)
            m._ran_pre_backward_hook.clear()
            m.training_state = TrainingState.IDLE
            for handle in m._handles:
                handle._training_state = HandleTrainingState.IDLE
            m._handles_prefetched.clear()
            if m._is_root:
                # reset this flag for cases like "one forward pass + multiple backward passes"
                self._post_backward_callback_queued = False

        if self._use_param_exec_order_policy() and self._param_exec_order_prep_stage:
            self._param_exec_order_policy_second_iter_init()

    def _param_exec_order_policy_second_iter_init(self) -> None:
        self._param_exec_order_prep_stage = False
        # Let the parameters in self._fsdp_params_exec_order ordered based on
        # the execution order in the forward pass.
        self._fsdp_params_exec_order.reverse()
        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                assert hasattr(
                    m, "_param_exec_order_policy"
                ), "Non-root FSDP modules should also have _param_exec_order_policy attribute"
                assert hasattr(
                    m, "_param_exec_order_prep_stage"
                ), "Non-root FSDP modules should also have _param_exec_order_prep_stage attribute"
                m._param_exec_order_prep_stage = False
        # TODO (linjianma): Construct a fsdp_wrap_map whose keys are all children modules with a FSDP wrap,
        # and values are its FSDP wraps. These children FSDP wraps will be detached from the root FSDP module
        # and will be used to schedule the parameters (rebuild_full_params and reshard).
        # TODO (linjianma): Remove all internal FSDP wraps from the root FSDP module.
        # TODO (linjianma): Based on self._fsdp_params_exec_order, get the information
        # needed to patch the forward() function of each key in the fsdp_wrap_map. The rules are as follows:
        # 1: Before each forward(), rebuild_full_params of all parameters that are currently sharded and
        # will be used in the forward, and reshard all parameters that are currently full and will not be
        # used in the next forward()
        # 2: After each forward(), reshard all parameters just used in the forward, and rebuild_full_params of
        # all parameters that will be used next.
        # TODO (linjianma): Patch the forward of each model in the keys
        # of fsdp_wrap_map based on the information above.

    def _assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across FSDP
        instances. Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`no_sync()` on inner FSDP instances is not supported. Please call `no_sync()` on root FSDP module."
            )
        self._assert_state(TrainingState.IDLE)
        old_flags = []
        for m in self.modules():
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m._sync_gradients))
                m._sync_gradients = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                assert not m._sync_gradients, (
                    "`_sync_gradients` was incorrectly set to "
                    "`True` while in the `no_sync()` context manager"
                )
                m._sync_gradients = old_flag

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ) -> torch.Tensor:
        """
        Clips the gradient norm of all parameters. The norm is computed over
        all parameters' gradients as viewed as a single vector, and the
        gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to ``torch.nn.utils.clip_grad_norm_`` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it for FSDP models would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`clip_grad_norm_()` should only be called on the root FSDP instance"
            )
        self._assert_state(TrainingState.IDLE)
        _wait_for_computation_stream(
            torch.cuda.current_stream(),
            self._streams["unshard"],
            self._streams["pre_unshard"],
        )

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        # Compute the local gradient norm (only including this rank's shard
        # of the gradients)
        local_norm = _get_grad_norm(self.parameters(), norm_type).to(
            self.compute_device
        )
        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = local_norm
            dist.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group
            )
        else:
            total_norm = local_norm**norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            total_norm = total_norm ** (1.0 / norm_type)
        if self.cpu_offload.offload_params:
            total_norm = total_norm.cpu()

        clip_coef = torch.tensor(
            max_norm, dtype=total_norm.dtype, device=total_norm.device
        ) / (total_norm + 1e-6)
        # Multiplying by the clamped coefficient is meaningless when it is
        # equal to 1, but it avoids the host-device sync that would result from
        # `if clip_coef < 1`
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        grads = [param.grad for param in self.parameters() if param.grad is not None]
        for grad in grads:
            grad.detach().mul_(clip_coef_clamped.to(grad.device))
        return total_norm

    @staticmethod
    def _warn_optim_input(optim_input):
        if optim_input is not None:
            warnings.warn(
                "The `optim_input` argument is deprecated and will be removed after PyTorch 1.13. You may remove it "
                "from your code without changing its functionality."
            )

    @staticmethod
    def _is_using_optim_input(optim_input, optim) -> bool:
        if optim_input is None and optim is None:
            # Use the default behavior of `optim_input``
            return True
        if optim_input is not None:
            # Use the `optim_input` code path
            return True
        # Use the `optim` code path
        return False

    @staticmethod
    def _raise_on_use_orig_params_optim_checkpoint(model: nn.Module):
        if any(
            fsdp_module._use_orig_params
            for fsdp_module in FullyShardedDataParallel.fsdp_modules(model)
        ):
            raise NotImplementedError(
                "Optimizer state checkpointing is not supported yet for `use_orig_params=True`"
            )

    @staticmethod
    def full_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        rank0_only: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in ``model`` are mapped back to their unflattened parameters.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications. However, if ``rank0_only=True``, then
            the state dict is only populated on rank 0, and all other ranks
            return an empty :class:`dict`.

        .. warning:: Unlike ``torch.optim.Optimizer.state_dict()``, this method
            uses full parameter names as keys instead of parameter IDs.

        .. note:: Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
            contained in the optimizer state dict are not cloned, so there may
            be aliasing surprises. For best practices, consider saving the
            returned optimizer state dict immediately, e.g. using
            ``torch.save()``.

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer ``optim`` representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            rank0_only (bool): If ``True``, saves the populated :class:`dict`
                only on rank 0; if ``False``, saves it on all ranks. (Default:
                ``True``)
            group (dist.ProcessGroup): Model's process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an empty :class:`dict`.
        """
        FullyShardedDataParallel._raise_on_use_orig_params_optim_checkpoint(model)
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=False,
            group=group,
            using_optim_input=using_optim_input,
        )

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        The API is similar to :meth:`full_optim_state_dict` but this API chunks
        all non-zero-dimension states to :class:`ShardedTensor` to save memory.
        This API should only be used when the model ``state_dict`` is derived
        with the context manager ``with state_dict_type(SHARDED_STATE_DICT):``.

        For the detailed usage, refer to :meth:`full_optim_state_dict`.

        .. warning:: The returned state dict contains ``ShardedTensor`` and
            cannot be directly used by the regular ``optim.load_state_dict``.
        """
        FullyShardedDataParallel._raise_on_use_orig_params_optim_checkpoint(model)
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        # TODO: The ultimate goal of the optimizer state APIs should be the same
        # as state_dict/load_state_dict -- using one API to get optimizer states
        # and one API to load optimizer states. ``state_dict_type`` will be used
        # to decide which optimizer states should be returned.
        # There are currently two APIs to load a full optimizer state. So the
        # first step of the unification is to merge the two full optimizer state
        # loading APIs.
        # Task: https://github.com/pytorch/pytorch/issues/82232
        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_input=optim_input,
            rank0_only=False,
            shard_state=True,
            group=group,
            using_optim_input=using_optim_input,
        )

    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Shards the full optimizer state dict ``full_optim_state_dict`` by
        remapping the state to flattened parameters instead of unflattened
        parameters and restricting to only this rank's part of the optimizer
        state. The first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)
            >>> torch.save(full_osd, PATH)
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim = ...
            >>> full_osd = torch.load(PATH)
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, new_model)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                full non-sharded optimizer state.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._raise_on_use_orig_params_optim_checkpoint(model)
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        sharded_osd = _flatten_optim_state_dict(
            full_optim_state_dict,
            model,
            True,
        )
        return _rekey_sharded_optim_state_dict(
            sharded_osd,
            model,
            optim,
            optim_input,
            using_optim_input,
        )

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:``shard_full_optim_state_dict``.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
        FullyShardedDataParallel._raise_on_use_orig_params_optim_checkpoint(model)
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        # TODO: The implementation is the same as ``shard_full_optim_state_dict``.
        # See the TODO in ``shard_full_optim_state_dict`` for the future
        # unification plan.
        flattened_osd = _flatten_optim_state_dict(
            sharded_optim_state_dict,
            model=model,
            shard_state=True,
        )
        return _rekey_sharded_optim_state_dict(
            flattened_osd,
            model,
            optim,
            optim_input,
            using_optim_input,
        )

    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: Optional[Dict[str, Any]],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        group: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Scatters the full optimizer state dict from rank 0 to all other ranks,
        returning the sharded optimizer state dict on each rank. The return
        value is the same as :meth:`shard_full_optim_state_dict`, and on rank
        0, the first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim, new_group = ...
            >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
                dict corresponding to the unflattened parameters and holding
                the full non-sharded optimizer state if on rank 0; the argument
                is ignored on nonzero ranks.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)
            group (dist.ProcessGroup): Model's process group or ``None`` if
                using the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._raise_on_use_orig_params_optim_checkpoint(model)
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        # Try to use the passed-in process group, the model's process group,
        # or the default process group (i.e. `None`) in that priority order
        if group is None and hasattr(model, "process_group"):
            group = model.process_group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        # Check for a valid broadcast device, preferring GPU when available
        using_nccl = dist.distributed_c10d._check_for_nccl_backend(group)
        broadcast_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if using_nccl and not torch.cuda.is_available():
            raise RuntimeError("NCCL requires a GPU for collectives")
        # Flatten the optimizer state dict and construct a copy with the
        # positive-dimension tensors' shapes in place of the tensors themselves
        # since those tensors will be broadcast separately to avoid copying
        if rank == 0:
            if full_optim_state_dict is None:
                raise ValueError("Rank 0 must pass in the full optimizer state dict")
            flat_osd = _flatten_optim_state_dict(
                full_optim_state_dict,
                model=model,
                shard_state=False,
            )
            processed_osd = _process_pos_dim_tensor_state(flat_osd, world_size)
        # Broadcast the optim state dict without positive-dimension tensor
        # state and the FSDP parameter IDs from rank 0 to all ranks
        processed_osd = _broadcast_processed_optim_state_dict(
            processed_osd if rank == 0 else None,
            rank,
            group,
        )
        # Broadcast positive-dimension tensor state (both sharded tensors for
        # FSDP parameters and unsharded tensors for non-FSDP parameters)
        sharded_osd = _broadcast_pos_dim_tensor_states(
            processed_osd,
            flat_osd if rank == 0 else None,
            rank,
            world_size,
            group,
            broadcast_device,
        )
        # Rekey the optimizer state dict to use parameter IDs according to this
        # rank's `optim`
        sharded_osd = _rekey_sharded_optim_state_dict(
            sharded_osd,
            model,
            optim,
            optim_input,
            using_optim_input,
        )
        return sharded_osd

    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: Dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Re-keys the optimizer state dict ``optim_state_dict`` to use the key
        type ``optim_state_key_type``. This can be used to achieve
        compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> osd = nonwrapped_optim.state_dict()
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
            >>> wrapped_model, wrapped_optim = ...
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
            >>> wrapped_optim.load_state_dict(sharded_osd)

        Returns:
            Dict[str, Any]: The optimizer state dict re-keyed using the
            parameter keys specified by ``optim_state_key_type``.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
        assert optim_state_key_type in (
            OptimStateKeyType.PARAM_NAME,
            OptimStateKeyType.PARAM_ID,
        )
        osd = optim_state_dict  # alias
        # Validate that the existing parameter keys are uniformly typed
        uses_param_name_mask = [type(param_key) is str for param_key in osd["state"]]
        uses_param_id_mask = [type(param_key) is int for param_key in osd["state"]]
        if (any(uses_param_name_mask) and not all(uses_param_name_mask)) or (
            any(uses_param_id_mask) and not all(uses_param_id_mask)
        ):
            error_msg = f"Invalid parameter keys: {osd['state'].keys()}"
            raise ValueError(error_msg)
        # Return directly if the existing key type matches the target key type
        if (
            optim_state_key_type == OptimStateKeyType.PARAM_NAME
            and all(uses_param_name_mask)
        ) or (
            optim_state_key_type == OptimStateKeyType.PARAM_ID
            and all(uses_param_id_mask)
        ):
            return osd
        # Otherwise, actually perform the re-keying
        new_osd = {}
        if optim_state_key_type == OptimStateKeyType.PARAM_NAME:  # ID -> name
            param_id_to_param = (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_id_to_param(optim)
            )
            param_to_param_name = _get_param_to_fqn(model)
            param_id_to_param_name: List[str] = [
                param_to_param_name[param] for param in param_id_to_param
            ]
            new_osd["state"] = {
                param_id_to_param_name[param_id]: param_state
                for param_id, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted(
                    [
                        param_id_to_param_name[param_id]
                        for param_id in param_group["params"]
                    ]
                )
            return new_osd
        elif optim_state_key_type == OptimStateKeyType.PARAM_ID:  # name -> ID
            param_name_to_param = _get_fqn_to_param(model)
            param_to_param_id = (
                _get_param_to_param_id_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_to_param_id(optim)
            )
            # Because not all model parameters may be passed as the optimizer
            # input, we may need to drop some parameters from this mapping
            param_name_to_param_id = {
                param_name: param_to_param_id[param]
                for param_name, param in param_name_to_param.items()
                if param in param_to_param_id
            }
            new_osd["state"] = {
                param_name_to_param_id[param_name]: param_state
                for param_name, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted(
                    [
                        param_name_to_param_id[param_name]
                        for param_name in param_group["params"]
                    ]
                )
            return new_osd
        return new_osd  # should never reach here

    def register_comm_hook(self, state: object, hook: callable):
        """
        Registers a communication hook which is an enhancement that provides a
        flexible hook to users where they can specify how FSDP aggregates gradients
        across multiple workers.
        This hook can be used to implement several algorithms like
        `GossipGrad <https://arxiv.org/abs/1803.05880>`_ and gradient compression
        which involve different communication strategies for
        parameter syncs while training with :class:`FullyShardedDataParallel`.

        .. warning ::
            FSDP communication hook should be registered before running an initial forward pass
            and only once.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in `GossipGrad <https://arxiv.org/abs/1803.05880>`_, etc.
                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable, which has one of the following signatures:
                            1) ``hook: Callable[torch.Tensor] -> None``:
                            This function takes in a Python tensor, which represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units).
                            It then performs all necessary processing and returns ``None``;
                            2) ``hook: Callable[torch.Tensor, torch.Tensor] -> None``:
                            This function takes in two Python tensors, the first one represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units). The latter
                            represents a pre-sized tensor to store a chunk of a sharded gradient after
                            reduction.
                            In both cases, callable performs all necessary processing and returns ``None``.
                            Callables with signature 1 are expected to handle gradient communication for a `NO_SHARD` case.
                            Callables with signature 2 are expected to handle gradient communication for sharded cases.

        """
        if not self.check_is_root():
            raise AssertionError(
                "register_comm_hook can only be called on a root instance."
            )
        for submodule in self.fsdp_modules(self):
            assert (
                not submodule._hook_registered
            ), "communication hook can be only registered once"
            submodule._hook_registered = True
            assert submodule._communication_hook == _get_default_comm_hook(
                self.sharding_strategy
            ), f"communication hook should be default, but it is {submodule._communication_hook.__name__} instead"
            submodule._communication_hook_state = state
            submodule._communication_hook = hook

    def _init_param_exec_order_wrap_policy(self, *args, **kwargs) -> None:
        auto_wrap_policy = kwargs["auto_wrap_policy"]
        module = kwargs["module"]
        assert hasattr(auto_wrap_policy, "tracing_config")
        if not _TORCH_FX_AVAIL:
            assert (
                auto_wrap_policy.tracing_config is None
            ), "tracing_config should be None when torch.fx is not enabled"
        elif isinstance(auto_wrap_policy.tracing_config, TracingConfig):
            tracer = auto_wrap_policy.tracing_config.tracer
            execution_info = _init_execution_info(module)

            for m in module.modules():
                assert not isinstance(
                    m, FullyShardedDataParallel
                ), "The input module of _patch_tracer should not contain FSDP modules"

            with _patch_tracer(
                tracer=tracer,
                root_module=module,
                execution_info=execution_info,
            ):
                try:
                    tracer.trace(module, auto_wrap_policy.tracing_config.concrete_args)
                except BaseException as e:
                    raise RuntimeError(
                        "tracer.trace failed inside _init_param_exec_order_wrap_policy"
                        f" with the error: {e}."
                    )
        else:
            assert (
                auto_wrap_policy.tracing_config is None
            ), "tracing_config should either be an instance of TracingConfig or be None"
        # The initial FSDP wrapping is done with auto_wrap_policy.init_policy
        kwargs["auto_wrap_policy"] = auto_wrap_policy.init_policy
        self.__init__(*args, **kwargs)
        self._param_exec_order_policy: bool = True
        # self._param_exec_order_prep_stage is set to True before we get the execution order
        self._param_exec_order_prep_stage: bool = True
        # A list that stores the flatten parameters and its name based on the parameter execution order
        self._fsdp_params_exec_order: List[FlatParameter] = []
        if _TORCH_FX_AVAIL and isinstance(
            auto_wrap_policy.tracing_config, TracingConfig
        ):
            # Initialize a dict that maps each module to its parent FSDP wrap
            module_to_fsdp: Dict[nn.Module, FullyShardedDataParallel] = dict()
            for wrap in self.fsdp_modules(self):
                module_to_fsdp[wrap.module] = wrap
            # Set self._fsdp_params_exec_order based on execution_info.module_forward_order.
            # TODO (linjianma): self._fsdp_params_exec_order will be set based on
            # the parameter execution order rather than module_forward_order,
            # once the non-recursive wrapping policy is fully implemented.
            for m in execution_info.module_forward_order:
                if m in module_to_fsdp:
                    for flat_param in module_to_fsdp[m].params:
                        self._fsdp_params_exec_order.append(flat_param)
            self._param_exec_order_prep_stage = False

        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                # Assignment by reference, so each children FSDP wrap has access to
                # the _fsdp_params_exec_order of the root module
                m._fsdp_params_exec_order = self._fsdp_params_exec_order
                m._param_exec_order_policy = self._param_exec_order_policy
                m._param_exec_order_prep_stage = self._param_exec_order_prep_stage

    def _use_param_exec_order_policy(self) -> bool:
        return (
            hasattr(self, "_param_exec_order_policy") and self._param_exec_order_policy
        )

    def _is_param_exec_order_prep_stage(self) -> bool:
        is_prep_stage = (
            hasattr(self, "_param_exec_order_prep_stage")
            and self._param_exec_order_prep_stage
        )
        if not is_prep_stage:
            for p in self.parameters():
                assert not hasattr(
                    p, "_params_exec_order_hook_handle"
                ), "When not in execution order prep stage, all _params_exec_order_hook_handle should be removed."
        return is_prep_stage


def _get_grad_norm(
    params: List[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Returns the gradient norm of parameters ``param`` s, where the gradients
    are viewed as a single vector.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)
    grads = [param.grad for param in params_with_grad]
    grad_dtypes = set(grad.dtype for grad in grads)
    if len(grad_dtypes) != 1:
        raise ValueError(
            f"Requires uniform dtype across all gradients but got {grad_dtypes}"
        )
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    grad_norm = grad_norm.to(grads[0].dtype)
    return grad_norm


def _get_param_to_fqn(
    model: torch.nn.Module,
) -> Dict[torch.nn.Parameter, str]:
    """
    Constructs a mapping from parameters to their parameter names. ``model``
    should not contain any :class:`FullyShardedDataParallel` instances, which
    means that none of the parameters should be ``FlatParameter`` s. As a
    result, compared to :meth:`_get_param_to_fqns`, the mapped
    values may be flattened from singleton :class:`list` s to the contained
    names themselves.

    Args:
        model (torch.nn.Module): Root module, which should not contain any
            :class:`FullyShardedDataParallel` instances.
    """
    param_to_param_names = _get_param_to_fqns(model)
    for param_names in param_to_param_names.values():
        assert len(param_names) > 0, (
            "`_get_param_to_fqns()` " "should not construct empty lists"
        )
        if len(param_names) > 1:
            raise RuntimeError(
                "Each parameter should only map to one parameter name but got "
                f"{len(param_names)}: {param_names}"
            )
    param_to_param_name = {
        param: param_names[0] for param, param_names in param_to_param_names.items()
    }
    return param_to_param_name


def _get_fqn_to_param(
    model: torch.nn.Module,
) -> Dict[str, torch.nn.Parameter]:
    """Constructs the inverse mapping of :meth:`_get_param_to_fqn`."""
    param_to_param_name = _get_param_to_fqn(model)
    return dict(zip(param_to_param_name.values(), param_to_param_name.keys()))
