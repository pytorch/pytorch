import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
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
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_param_to_fqns,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    TrainingState,
)
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)
from torch.distributed.fsdp._runtime_utils import (
    _get_fsdp_root_states,
    _is_fsdp_root,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.utils import _p_assert

from ._optim_utils import (
    _flatten_optim_state_dict,
    _get_param_id_to_param_from_optim_input,
    _get_param_key_to_param,
    _get_param_to_param_id_from_optim_input,
    _get_param_to_param_key,
    _optim_state_dict,
    _rekey_sharded_optim_state_dict,
)
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
    _deregister_orig_params,
    _register_flat_param,
    _register_orig_params,
    _unshard_params,
    _unshard_params_recurse,
)
from .flat_param import FlatParameter
from .wrap import ModuleWrapPolicy


__all__ = [
    "FullyShardedDataParallel",
    "OptimStateKeyType",
]


FLAT_PARAM = "_flat_param"


class OptimStateKeyType(Enum):
    PARAM_NAME = auto()
    PARAM_ID = auto()


class FullyShardedDataParallel(nn.Module, _FSDPState):
    """
    A wrapper for sharding module parameters across data parallel workers. This
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
        The optimizer must be initialized *after* the module has been wrapped
        with FSDP since FSDP will shard and transform the module's parameters
        in a way that may not preserve the original parameter variables. Thus,
        the previously initialized optimizer may have stale references to the
        parameters.

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
        Passing in the ``sync_module_states=True`` flag requires ``module`` to
        be on GPU or to use the ``device_id`` argument to specify a CUDA device
        that FSDP will move ``module`` to in the FSDP constructor. This is
        because ``sync_module_states=True`` requires GPU communication.

    .. warning::
        As of PyTorch 1.12, FSDP only offers limited support for shared parameters
        (for example, setting one ``Linear`` layer's weight to another's). In
        particular, modules that share parameters must be wrapped as part of the
        same FSDP unit. If enhanced shared parameter support is needed for your
        use case, please ping https://github.com/pytorch/pytorch/issues/77724

    .. warning::
        FSDP has some constraints on freezing parameters (i.e. setting
        ``param.requires_grad=False``). For ``use_orig_params=False``, each
        FSDP instance must manage parameters that are all frozen or all
        non-frozen. For ``use_orig_params=True``, FSDP supports mixing frozen
        and non-frozen, but we recommend not doing so since then the gradient
        memory usage will be higher than expected (namely, equivalent to not
        freezing those parameters). This means that ideally, frozen parameters
        should be isolated into their own ``nn.Module`` s and wrapped
        separately with FSDP.

    .. note::
        Attempting to run the forward pass of a submodule that is contained in an
        FSDP instance is not supported and will result in errors. This is because the
        submodule's parameters will be sharded, but it itself is not an FSDP instance,
        so its forward pass will not all-gather the full parameters appropriately.
        This could potentially happen when attempting to run only the encoder of a
        encoder-decoder model, and the encoder is not wrapped in its own FSDP instance. To
        resolve this, please wrap the submodule in its own FSDP unit.

    .. note::
        FSDP moves input tensors to the ``forward`` method to the GPU compute
        device, so the user does not need to manually move them from CPU.

    .. warning::
        The user should not modify the parameters between forward and backward
        without using the :meth:`summon_full_params` context since the
        modifications may not persist. Moreover, for ``use_orig_params=False``,
        accessing the original parameters between forward and backward may
        raise an illegal memory access.

    .. warning::
        For ``use_orig_params=True``, ``ShardingStrategy.SHARD_GRAD_OP``
        exposes the unsharded parameters, not the sharded parameters, after
        forward since it does not free the unsharded ones, unlike
        ``ShardingStrategy.FULL_SHARD``. One caveat is that, since gradients
        are always sharded or ``None``, ``ShardingStrategy.SHARD_GRAD_OP`` will
        not expose the sharded gradients with the unsharded parameters after
        forward. If you want to inspect the gradients, try
        :meth:`summon_full_params` with ``with_grads=True``.

    .. warning::
        FSDP replaces managed modules' parameters with ``torch.Tensor`` views
        during forward and backward computation for autograd-related reasons.
        If your module's forward relies on saved references to the parameters
        instead of reacquiring the references each iteration, then it will not
        see FSDP's newly created views, and autograd will not work correctly.

    .. note::
        With ``limit_all_gathers=True``, you may see a gap in the FSDP
        pre-forward where the CPU thread is not issuing any kernels. This is
        intentional and shows the rate limiter in effect. Synchronizing the CPU
        thread in that way prevents over-allocating memory for subsequent
        all-gathers, and it should not actually delay GPU kernel execution.

    .. note::
        When using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD`` with the
        sharding process group being intra-node and the replication process
        group being inter-node, setting ``NCCL_CROSS_NIC=1`` can help improve
        the all-reduce times over the replication process group for some
        cluster setups.

    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]):
            This is the process group over which the model is sharded and thus
            the one used for FSDP's all-gather and reduce-scatter collective
            communications. If ``None``, then FSDP uses the default process
            group. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a tuple of
            process groups, representing the groups over which to shard and
            replicate, respectively. If ``None``, then FSDP constructs process
            groups for the user to shard intra-node and replicate inter-node.
            (Default: ``None``)
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy, which may trade off memory
            saving and communication overhead. See :class:`ShardingStrategy`
            for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], ModuleWrapPolicy]]):
            This specifies a policy to apply FSDP to submodules of ``module``,
            which is needed for communication and computation overlap and thus
            affects performance. If ``None``, then FSDP only applies to
            ``module``, and users should manually apply FSDP to parent modules
            themselves (proceeding bottom-up). For convenience, this accepts
            ``ModuleWrapPolicy`` directly, which allows users to specify the
            module classes to wrap (e.g. the transformer block). Otherwise,
            this should be a callable that takes in three arguments
            ``module: nn.Module``, ``recurse: bool``, and
            ``nonwrapped_numel: int`` and should return a ``bool`` specifying
            whether the passed-in ``module`` should have FSDP applied if
            ``recurse=False`` or if the traversal should continue into the
            module's subtree if ``recurse=True``. Users may add additional
            arguments to the callable. The ``size_based_auto_wrap_policy`` in
            ``torch.distributed.fsdp.wrap.py`` gives an example callable that
            applies FSDP to a module if the parameters in its subtree exceed
            100M numel. We recommend printing the model after applying FSDP
            and adjusting as needed.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     nonwrapped_numel: int,
                >>>     # Additional custom arguments
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return nonwrapped_numel >= min_num_params
                >>> # Configure a custom `min_num_params`
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. If
            ``None``, then FSDP does not backward prefetch, and there is no
            communication and computation overlap in the backward pass. See
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
            specifies how modules that are currently on the meta device should
            be initialized onto an actual device. As of v1.12, FSDP detects
            modules with parameters or buffers on meta device via ``is_meta``
            and either applies ``param_init_fn`` if specified or calls
            ``nn.Module.reset_parameters()`` otherwise. For both cases, the
            implementation should *only* initialize the parameters/buffers of
            the module, not those of its submodules. This is to avoid
            re-initialization. In addition, FSDP also supports deferred
            initialization via torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init()`` API, where the deferred modules are initialized
            by calling ``param_init_fn`` if specified or torchdistX's default
            ``materialize_module()`` otherwise. If ``param_init_fn`` is
            specified, then it is applied to all meta-device modules, meaning
            that it should probably case on the module type. FSDP calls the
            initialization function before parameter flattening and sharding.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module: nn.Module):
                >>>     # E.g. initialize depending on the module type
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or
            ``torch.device`` giving the CUDA device on which FSDP
            initialization takes place, including the module initialization
            if needed and the parameter sharding. This should be specified to
            improve initialization speed if ``module`` is on CPU. If the
            default CUDA device was set (e.g. via ``torch.cuda.set_device``),
            then the user may pass ``torch.cuda.current_device`` to this.
            (Default: ``None``)
        sync_module_states (bool): If ``True``, then each FSDP module will
            broadcast module parameters and buffers from rank 0 to ensure that
            they are replicated across ranks (adding communication overhead to
            this constructor). This can help load ``state_dict`` checkpoints
            via ``load_state_dict`` in a memory efficient way. See
            :class:`FullStateDictConfig` for an example of this. (Default:
            ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next forward-pass all-gather before the current forward
            computation. This is only useful for CPU-bound workloads, in which
            case issuing the next all-gather earlier may improve overlap. This
            should only be used for static-graph models since the prefetching
            follows the first iteration's execution order. (Default: ``False``)
        limit_all_gathers (bool): If ``True``, then FSDP explicitly
            synchronizes the CPU thread to ensure GPU memory usage from only
            *two* consecutive FSDP instances (the current instance running
            computation and the next instance whose all-gather is prefetched).
            If ``False``, then FSDP allows the CPU thread to issue all-gathers
            without any extra synchronization. (Default: ``True``) We often
            refer to this feature as the "rate limiter". This flag should only
            be set to ``False`` for specific CPU-bound workloads with low
            memory pressure in which case the CPU thread can aggressively issue
            all kernels without concern for the GPU memory usage.
        use_orig_params (bool): Setting this to ``True`` has FSDP use
            ``module`` 's original parameters. FSDP exposes those original
            parameters to the user via :meth:`nn.Module.named_parameters`
            instead of FSDP's internal :class:`FlatParameter` s. This means
            that the optimizer step runs on the original parameters, enabling
            per-original-parameter hyperparameters. FSDP preserves the original
            parameter variables and manipulates their data between unsharded
            and sharded forms, where they are always views into the underlying
            unsharded or sharded :class:`FlatParameter`, respectively. With the
            current algorithm, the sharded form is always 1D, losing the
            original tensor structure. An original parameter may have all,
            some, or none of its data present for a given rank. In the none
            case, its data will be like a size-0 empty tensor. Users should not
            author programs relying on what data is present for a given
            original parameter in its sharded form. ``True`` is required to
            use ``torch.compile()``. Setting this to ``False`` exposes FSDP's
            internal :class:`FlatParameter` s to the user via
            :meth:`nn.Module.named_parameters`. (Default: ``False``)
        ignored_states (Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]):
            Ignored parameters or modules that will not be managed by this FSDP
            instance, meaning that the parameters are not sharded and their
            gradients are not reduced across ranks. This argument unifies with
            the existing ``ignored_modules`` argument, and we may deprecate
            ``ignored_modules`` soon. For backward compatibility, we keep both
            ``ignored_states`` and `ignored_modules``, but FSDP only allows one
            of them to be specified as not ``None``.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Union[Callable, ModuleWrapPolicy]] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        _init_ignored_module_states(self, module, ignored_modules, ignored_states)
        _init_device_handle(self, module, self._ignored_params, device_id)

        # Add module annotations for Dynamo support (see function for details)
        _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

        # Initializes self.process_group, along with rank and world size. This will
        # also set another attribute, _inter_node_pg, to control the process group
        # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
        # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
        # the same process group state as the root FSDP module.
        self.device_mesh = device_mesh
        _init_process_group_state(
            self,
            process_group,
            sharding_strategy,
            auto_wrap_policy,
            device_mesh,
        )
        if auto_wrap_policy is not None:
            root_kwargs = {
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
                "ignored_states": self._ignored_params,
            }
            if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
                # Share root process groups with children to maintain
                # the invariant that all FSDP modules will have the same
                # process groups.
                root_kwargs["process_group"] = (self.process_group, self._inter_node_pg)

            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                root_kwargs,
                FullyShardedDataParallel,
            )

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
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            _register_flat_param(self, self)

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)
        _register_all_state_dict_hooks(self)

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
        return hasattr(self, "_handle") and self._handle is not None

    @property
    def _flat_param(self) -> Optional[FlatParameter]:
        return self._handle.flat_param if self._handle else None

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
        return _is_fsdp_root(self, self)

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
        if root_only:
            return _get_fsdp_root_states(module)
        return traversal_utils._get_fsdp_states(module)

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
        # Use `_unshard_params_recurse()` with `recurse=False` instead of
        # `_unshard_fsdp_state_params()` directly to perform lazy
        # initialization, which is needed to initialize `FlatParameter`
        # parameter attributes as required by the unshard logic
        with _unshard_params_recurse(
            self,
            self,
            recurse=False,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False,
        ):
            ret = super().apply(fn)

        # Reset lazy init called in `_unshard_params_recurse()` since `apply()`
        # may have been called on FSDP instance that is not truly a root, in
        # which case it will be incorrectly marked as one.
        if uninitialized and self._is_root:
            for module in traversal_utils._get_fsdp_states(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """
        Returns if the user explicitly enabled buffer mixed precision.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _low_precision_hook_enabled(self) -> bool:
        """
        Whether a low precision hook is registered or not.
        """
        return self._comm_hook is not None and self._comm_hook in LOW_PRECISION_HOOKS

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
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
    ) -> StateDictSettings:
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
            >>>     state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
            >>>     optim_state_dict_config = OptimStateDictConfig(offload_to_cpu=True),
            >>> )
            >>> param_state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the configuration
                for the optimizer state dict.

        Returns:
            A StateDictSettings that include the previous state_dict type and
            configuration for the module.
        """
        _state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }
        _optim_state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig,
        }

        # Use the default config if a state_dict config is not set.
        state_dict_config_type = _state_dict_type_to_config[state_dict_type]
        optim_state_dict_config_type = _optim_state_dict_type_to_config[state_dict_type]
        if state_dict_config is None:
            state_dict_config = state_dict_config_type()
        if optim_state_dict_config is None:
            optim_state_dict_config = optim_state_dict_config_type()
        if state_dict_config_type != type(state_dict_config):
            raise RuntimeError(
                f"Expected state_dict_config of type {state_dict_config_type} "
                f"but got {type(state_dict_config)}"
            )
        if optim_state_dict_config_type != type(optim_state_dict_config):
            raise RuntimeError(
                f"Expected optim_state_dict_config of type {optim_state_dict_config_type} "
                f"but got {type(optim_state_dict_config)}"
            )

        # Set the state_dict type and configurations.
        prev_state_dict_type = None
        prev_state_dict_config = None
        prev_optim_state_dict_config = None
        for submodule in traversal_utils._get_fsdp_states(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = submodule._state_dict_type
            else:
                assert (
                    prev_state_dict_type == submodule._state_dict_type
                ), "All FSDP modules should have the same state_dict_type."
            if prev_state_dict_config is None:
                prev_state_dict_config = submodule._state_dict_config
            else:
                assert isinstance(
                    submodule._state_dict_config, type(prev_state_dict_config)
                ), "All FSDP modules must have the same type of state_dict_config."
            if prev_optim_state_dict_config is None:
                prev_optim_state_dict_config = submodule._optim_state_dict_config
            else:
                assert isinstance(
                    submodule._optim_state_dict_config,
                    type(prev_optim_state_dict_config),
                ), "All FSDP modules must have the same type of optim_state_dict_config."

            submodule._state_dict_type = state_dict_type
            submodule._state_dict_config = state_dict_config
            submodule._optim_state_dict_config = optim_state_dict_config

        return StateDictSettings(
            prev_state_dict_type, prev_state_dict_config, prev_optim_state_dict_config
        )

    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings:
        """
        Get the state_dict_type and the corresponding configurations
        for the FSDP modules rooted at ``module``. The target module
        does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
        state_dict_settings: Optional[StateDictSettings] = None
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if state_dict_settings is None:
                state_dict_settings = StateDictSettings(
                    state_dict_type=submodule._state_dict_type,
                    state_dict_config=submodule._state_dict_config,
                    optim_state_dict_config=submodule._optim_state_dict_config,
                )
            else:
                submodule_settings = StateDictSettings(
                    submodule._state_dict_type,
                    submodule._state_dict_config,
                    submodule._optim_state_dict_config,
                )
                assert state_dict_settings == submodule_settings, (
                    "All FSDP modules must have the same state dict settings."
                    f"Got {submodule_settings} and {state_dict_settings}."
                )

            # If device_mesh is passed in when initalizing FSDP, we automatically turn the
            # _use_dtensor flag to be true for ShardedOptimStateDictConfig().
            if (
                getattr(module, "device_mesh", None)
                and state_dict_settings.state_dict_type
                == StateDictType.SHARDED_STATE_DICT
            ):
                state_dict_settings.optim_state_dict_config._use_dtensor = True
        return state_dict_settings

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
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
            state_dict_config (Optional[StateDictConfig]): the model ``state_dict``
                configuration for the target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the optimizer
               ``state_dict`` configuration for the target ``state_dict_type``.
        """
        prev_state_dict_settings = FullyShardedDataParallel.set_state_dict_type(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        )
        yield
        FullyShardedDataParallel.set_state_dict_type(
            module,
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
            prev_state_dict_settings.optim_state_dict_config,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the forward pass for the wrapped module, inserting FSDP-specific
        pre- and post-forward sharding logic.
        """
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module: nn.Module,
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
        with _unshard_params(
            module, recurse, writeback, rank0_only, offload_to_cpu, with_grads
        ):
            yield

    @contextlib.contextmanager
    def _deregister_orig_params_ctx(self):
        """
        This deregisters the original parameters and exposes the
        :class:`FlatParameter` s. If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method should
        only be called when using the original parameters.
        """
        _p_assert(
            self._use_orig_params,
            "`_deregister_orig_params_ctx()` should only be called when "
            "`_use_orig_params=True`",
        )
        for fsdp_module in traversal_utils._get_fsdp_states(self):
            _deregister_orig_params(fsdp_module, fsdp_module)
        try:
            yield
        finally:
            for fsdp_module in traversal_utils._get_fsdp_states(self):
                _register_orig_params(fsdp_module, fsdp_module)

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
        context = (
            self._deregister_orig_params_ctx()
            if self._use_orig_params
            else contextlib.nullcontext()
        )
        with context:
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
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
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
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

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

        .. note:: If every FSDP instance uses ``NO_SHARD``, meaning that no
            gradients are sharded across ranks, then you may directly use
            :func:`torch.nn.utils.clip_grad_norm_`.

        .. note:: If at least some FSDP instance uses a sharded strategy (i.e.
            one other than ``NO_SHARD``), then you should use this method
            instead of :func:`torch.nn.utils.clip_grad_norm_` since this method
            handles the fact that gradients are sharded across ranks.

        .. note:: The total norm returned will have the "largest" dtype across
            all parameters/gradients as defined by PyTorch's type promotion
            semantics. For example, if *all* parameters/gradients use a low
            precision dtype, then the returned norm's dtype will be that low
            precision dtype, but if there exists at least one parameter/
            gradient using FP32, then the returned norm's dtype will be FP32.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`clip_grad_norm_()` should only be called on the root FSDP instance"
            )
        self._assert_state(TrainingState.IDLE)
        # If every FSDP instance uses `NO_SHARD`, then we can directly use
        # the normal `nn.utils` one targeting local gradients
        all_no_shard = all(
            not handle.uses_sharded_strategy for handle in self._all_handles
        )
        if all_no_shard:
            return torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm, norm_type
            )
        # Otherwise, there exists some FSDP instance using a sharded strategy,
        # where sharded and non-sharded parameters must be handled separately
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        sharded_params = set()
        nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
        grads: List[torch.Tensor] = []
        for handle in self._all_handles:
            target_set = (
                sharded_params if handle.uses_sharded_strategy else nonsharded_params
            )
            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    target_set.add(param)
                    if param.grad is not None:
                        grads.append(param.grad)
            else:
                target_set.add(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)
        for param in self.parameters():
            not_fsdp_managed = (
                param not in sharded_params and param not in nonsharded_params
            )
            if not_fsdp_managed:
                nonsharded_params.add(param)
                if param.grad is not None:
                    grads.append(param.grad)
        # Compute local norms (forced to be in FP32)
        local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
            self.compute_device
        )
        local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
            self.compute_device
        )
        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
            dist.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group
            )
        else:
            total_norm = local_sharded_norm**norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            # All-reducing the local non-sharded norm would count it an extra
            # world-size-many times
            total_norm += local_nonsharded_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        if self.cpu_offload.offload_params:
            total_norm = total_norm.cpu()

        clip_coef = max_norm / (total_norm + 1e-6)
        # Multiplying by the clamped coefficient is meaningless when it is
        # equal to 1, but it avoids the host-device sync that would result from
        # `if clip_coef < 1`
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for grad in grads:
            grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))
        # Use the "largest" dtype by type promotion semantics to use the same
        # dtype as if we did not force local norm computation to be in FP32
        if len(grads) == 0:
            # If this rank has no gradients, then we must default to FP32
            # unless we use additional communication, which we prefer to avoid
            # since `clip_grad_norm_()` is called in the training loop
            warnings.warn(
                f"Called FSDP.clip_grad_norm_() on rank {self.rank} with no "
                "gradients -- returning the total norm in the default dtype "
                f"{total_norm.dtype}"
            )  # warn since this is generally unexpected
            return total_norm
        total_norm_dtype = functools.reduce(
            lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
            [grad.dtype for grad in grads],
        )
        return total_norm.to(total_norm_dtype)

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
    def _warn_legacy_optim_state_dict(curr: str, new: str):
        warnings.warn(
            f"``FullyShardedDataParallel.{curr}``is being deprecated and is "
            f"replaced by ``FullyShardedDataParallel.{new}``. "
            f"``FullyShardedDataParallel.{curr}`` may be removed after PyTorch 2.2."
        )

    @staticmethod
    def _optim_state_dict_impl(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Dict[str, Any],
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        rank0_only: bool = True,
        full_state_dict: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        The internal API that is used by all the optim_state_dict implementations.
        Given model, optim, the original optim_state_dict, this API removes the
        FSDP internal information and internal sharding from the optim_state_dict.
        """
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(optim_input)
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            using_optim_input = False
            assert optim_input is None and not rank0_only

        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        assert all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ), "Not all FSDP modules have the same _use_orig_params value"

        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=not full_state_dict,
            group=group,
            using_optim_input=using_optim_input,
            use_orig_params=use_orig_params,
        )

    @staticmethod
    def _optim_state_dict_to_load_impl(
        optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        full_state_dict: bool = True,
        rank0_only: bool = False,
        is_named_optimizer: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        The internal API that is used by all the load optim_state_dict implementations.
        Given model, optim, and the saved optim_state_dict, this API adds the FSDP
        internal information and internal sharding to the optim_state_dict.
        """
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(optim_input)
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            using_optim_input = False
            assert optim_input is None and not rank0_only

        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        assert all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ), "Not all FSDP modules have the same _use_orig_params value"

        if rank0_only and dist.get_rank(group) > 0:
            optim_state_dict = {}
        sharded_osd = _flatten_optim_state_dict(
            optim_state_dict,
            model=model,
            use_orig_params=use_orig_params,
            optim=(optim if is_named_optimizer else None),
            rank0_only=rank0_only,
            group=group,
        )
        return _rekey_sharded_optim_state_dict(
            sharded_osd,
            model=model,
            optim=optim,
            optim_input=optim_input,
            using_optim_input=using_optim_input,
            is_named_optimizer=is_named_optimizer,
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
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "full_optim_state_dict", "optim_state_dict"
        )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim.state_dict(),
            optim_input=optim_input,
            rank0_only=rank0_only,
            group=group,
            full_state_dict=True,
        )

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
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
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "sharded_optim_state_dict", "optim_state_dict"
        )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim.state_dict(),
            optim_input=None,
            rank0_only=False,
            full_state_dict=False,
            group=group,
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
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "shard_full_optim_state_dict", "optim_state_dict_to_load"
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=full_optim_state_dict,
            model=model,
            optim_input=optim_input,
            optim=optim,
            full_state_dict=True,
            is_named_optimizer=False,
        )

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
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
                Refer to :meth:`shard_full_optim_state_dict`.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "flatten_sharded_optim_state_dict", "optim_state_dict_to_load"
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=sharded_optim_state_dict,
            model=model,
            optim_input=None,
            optim=optim,
            full_state_dict=False,
            is_named_optimizer=False,
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
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "scatter_full_optim_state_dict", "optim_state_dict_to_load"
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=full_optim_state_dict,
            model=model,
            optim_input=optim_input,
            optim=optim,
            full_state_dict=True,
            rank0_only=True,
            is_named_optimizer=False,
            group=group,
        )

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
                else _get_param_key_to_param(optim)
            )
            param_to_param_name = _get_param_to_fqn(model)
            param_id_to_param_name: List[str] = [
                param_to_param_name[param] for param in param_id_to_param.values()
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
                else _get_param_to_param_key(optim)
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

    @staticmethod
    def optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Optional[Dict[str, Any]] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        Transforms the state_dict of ``optim`` for the ``model`` that is sharded
        by FSDP to one of the three types: 1) full optimizer state_dict, 2)
        sharded optimizer state_dict, 3) local optimizer state_dict.

        For full optimizer state_dict, all states are unflattened and not sharded.
        Rank0 only and CPU only can be specified via :meth:`state_dict_type` to
        avoid OOM.

        For sharded optimizer state_dict, all states are unflattened but sharded.
        CPU only can be specified via :meth:`state_dict_type` to further save
        memory.

        For local state_dict, no transformation will be performed. But a state
        will be converted from nn.Tensor to ShardedTensor to represent its sharding
        nature (this is not supported yet).

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     optim_state_dict, model, optim
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_state_dict (Dict[str, Any]): the target optimizer state_dict to
                transform. If the value is None, optim.state_dict() will be used. (
                Default: ``None``)
            group (dist.ProcessGroup): Model's process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model``. The sharding of the optimizer state is based on
            ``state_dict_type``.
        """
        state_dict_settings = FullyShardedDataParallel.get_state_dict_type(model)
        if optim_state_dict is None:
            optim_state_dict = optim.state_dict()
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            optim_input=None,
            rank0_only=getattr(
                state_dict_settings.optim_state_dict_config, "rank0_only", False
            ),
            full_state_dict=state_dict_settings.state_dict_type
            == StateDictType.FULL_STATE_DICT,
            group=group,
        )

    @staticmethod
    def optim_state_dict_to_load(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Dict[str, Any],
        is_named_optimizer: bool = False,
        load_directly: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        Given a ``optim_state_dict`` that is transformed through
        :meth:`optim_state_dict`, converts it to the flattened optimizer
        state_dict that can be loaded to ``optim`` which is the optimizer for
        ``model``.  ``model`` must be sharded by FullyShardedDataParallel.

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> original_osd = optim.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(
            >>>     model,
            >>>     optim,
            >>>     optim_state_dict=original_osd
            >>> )
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     model, optim, optim_state_dict
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_state_dict (Dict[str, Any]): The optimizer states to be loaded.
            is_named_optimizer (bool): Is this optimizer a NamedOptimizer or
                KeyedOptimizer. Only set to True if ``optim`` is TorchRec's
                KeyedOptimizer or torch.distributed's NamedOptimizer.
            load_directly (bool): If this is set to True, this API will also
                call optim.load_state_dict(result) before returning the result.
                Otherwise, users are responsible to call ``optim.load_state_dict()``
                (Default: ``False``)
            group (dist.ProcessGroup): Model's process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)
        """
        state_dict_settings = FullyShardedDataParallel.get_state_dict_type(model)
        result = FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=optim_state_dict,
            model=model,
            optim_input=None,
            optim=optim,
            full_state_dict=(
                state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT
            ),
            rank0_only=getattr(
                state_dict_settings.optim_state_dict_config, "rank0_only", False
            ),
            is_named_optimizer=is_named_optimizer,
            group=group,
        )
        if load_directly:
            optim.load_state_dict(result)
        return result

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
        for fsdp_state in traversal_utils._get_fsdp_states(self):
            if fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
                raise AssertionError(
                    f"Communication hook is not supported for hybrid strategies: {fsdp_state.sharding_strategy}"
                )
            if fsdp_state._comm_hook is not None:
                raise AssertionError("A communication hook is already registered")
            if not callable(hook):
                raise ValueError(
                    f"The communication hook must be callable but got {hook}"
                )
            fsdp_state._comm_hook = hook
            fsdp_state._comm_hook_state = state


def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Returns the gradient norm of parameters ``param`` s, where the gradients
    are viewed as a single vector. The returned norm is in FP32 even if
    parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)
    grads = [param.grad for param in params_with_grad]
    grad_dtypes = {grad.dtype for grad in grads}
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
        assert (
            len(param_names) > 0
        ), "`_get_param_to_fqns()` should not construct empty lists"
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
