# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    cast,
    NoReturn,
    Optional,
    overload,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import deprecated

import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed.utils import _get_root_modules

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo
from ._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import _get_module_fsdp_state, FSDPState


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.distributed.tensor import DeviceMesh, Shard

__all__ = [
    "fully_shard",
    "FSDPModule",
    "UnshardHandle",
    "register_fsdp_forward_method",
]


cls_to_fsdp_cls: dict[type, type] = {}


@overload
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = ...,
    reshard_after_forward: Union[bool, int] = ...,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: Optional[set[nn.Parameter]] = ...,
) -> FSDPModule: ...


@overload
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: Optional[DeviceMesh] = ...,
    reshard_after_forward: Union[bool, int] = ...,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: Optional[set[nn.Parameter]] = ...,
) -> list[FSDPModule]: ...


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
# [1] Python runtime decorator does not play well with static type checking
# so suppressing some type checks to support type overloads
# such that caller can still get correct return types based on input type
@contract(state_cls=FSDPState)  # type: ignore[misc] # see [1]
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Optional[Union[bool, int]] = None,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
):
    """
    Apply fully sharded data parallelism (FSDP) to ``module``, where FSDP
    shards module parameters, gradients, and optimizer states across data
    parallel workers to save memory at the cost of communication.

    At initialization, FSDP shards the module's parameters across the data
    parallel workers given by ``mesh``. Before forward, FSDP all-gathers the
    sharded parameters across the data-parallel workers to get the unsharded
    parameters for forward computation. If ``reshard_after_forward`` is
    ``True``, then FSDP frees the unsharded parameters after forward and
    re-all-gathers them in backward before gradient computation. After gradient
    computation, FSDP frees the unsharded parameters and reduce-scatters the
    unsharded gradients across data-parallel workers.

    This implementation represents the sharded parameters as :class:`DTensor` s
    sharded on dim-0, while the unsharded parameters will be like the original
    parameters on ``module`` (e.g. :class:`torch.Tensor` if originally
    :class:`torch.Tensor`). A module
    `forward pre-hook <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook>`_
    on ``module`` all-gathers the parameters, and a module
    `forward hook <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>`_
    on ``module`` frees them (if needed). Similar backward hooks all-gather
    parameters and later free parameters and reduce-scatter gradients.

    Since grouping multiple tensors together for one collective is critical for
    communication efficiency, this implementation makes this grouping first
    class. Calling :meth:`fully_shard` on ``module`` constructs one group that
    includes the parameters in ``module.parameters()`` except those already
    assigned to a group from an earlier call on a submodule. This means that
    :meth:`fully_shard` should be called bottom-up on your model. Each group's
    parameters are all-gathered in one collective, and its gradients are
    reduce-scattered in one collective. Partitioning the model into multiple
    groups ("layer by layer") allows for peak memory savings and communication/computation
    overlap. Users generally should *not* call :meth:`fully_shard` only on the
    topmost root module.

    Args:
        module (Union[nn.Module, List[nn.Module]): The module or modules to
            shard with FSDP and group together for communication.
        mesh (Optional[DeviceMesh]): This data parallel mesh defines the
            sharding and device. If 1D, then parameters are fully sharded
            across the 1D mesh (FSDP) with ``(Shard(0),)`` placement. If 2D,
            then parameters are sharded across the 1st dim and replicated
            across the 0th dim (HSDP) with ``(Replicate(), Shard(0))``
            placement. The mesh's device type gives the device type used for
            communication; if a CUDA or CUDA-like device type, then we use the
            current device.
        reshard_after_forward (Optional[Union[bool, int]]): This controls the parameter
            behavior after forward and can trade off memory and communication:

            - If ``True``, then this reshards parameters after forward and
              re-all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
              after forward and avoids the all-gather in backward. For best performance,
              we usually set ``False`` for the root module, because the root module
              is typically required immediately when the backward pass begins.
            - If ``None``, it is set to ``True`` for non-root modules and ``False``
              for root modules.
            - If an ``int``, then this represents the world size to reshard to
              after forward. It should be a non-trivial divisor of the ``mesh``
              shard dim size (i.e. excluding 1 and the dim size itself). A
              choice may be the intra-node size (e.g. ``torch.cuda.device_count()``).
              This allows the all-gather in backward to be over a smaller world
              size at the cost of higher memory usage than setting to ``True``.
            - After forward, the parameters registered to the module depend on
              to this: The registered parameters are the sharded parameters if
              ``True``; unsharded parameters if ``False``; and the parameters
              resharded to the smaller mesh otherwise. To modify the parameters
              between forward and backward, the registered parameters must be
              the sharded parameters. For ``False`` or an ``int``, this can be
              done by manually resharding via :meth:`reshard`.
        shard_placement_fn (Optional[Callable[[nn.Parameter], Optional[Shard]]]):
            This callable can be used to override the sharding placement for a
            parameter to shard a parameter on a dimension other than dim-0. If
            this callable returns a :class:`Shard` placement (not ``None``),
            then FSDP will shard according to that placement (e.g. ``Shard(1)``).
            If sharding on a nonzero dim, we currently require even sharding,
            i.e. the tensor dim size on that dim must be divisible by the FSDP
            shard mesh size.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
        ignored_params: Optional(Set[nn.Parameter]): The set of parameters to be
            ignored by FSDP. They will not be sharded, nor moved to the device
            during init, nor have their gradients reduced in backward.

    Returns:
        FSDPModule: The module with FSDP applied (in-place).
    """
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    auto_reshard_after_forward = reshard_after_forward is None
    # If the user does not provide ``reshard_after_forward``, we set it to True.
    # During lazy_init, we identify which module is the root and override its value to False
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward if not auto_reshard_after_forward else True,  # type: ignore[arg-type]
        mesh_info,
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined] # see [1]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


class FSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `FSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, freeing the unsharded parameters if
        they are allocated and registering the sharded parameters to the
        module. This method is *not* recursive.
        """
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> Optional[UnshardHandle]:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive. The unshard follows the
        :class:`MixedPrecisionPolicy`, so it will all-gather following
        ``param_dtype`` if set.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then FSDP will wait on the pending
            unshard in the module's pre-forward for the user. The user only
            needs to call :meth:`wait` explicitly if the wait should happen
            before pre-forward.
        """
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(fsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together. This is the equivalence of
        `no_sync` in FSDP1.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after forward. This can be
        used to change the ``reshard_after_forward`` FSDP arg at runtime. For
        example, this can be used to set the FSDP root module's value to
        ``True`` (since it is otherwise specially set to ``False``), or it can
        set an FSDP module's value to ``False`` for running evals and set back
        to ``True`` for training.

        Args:
            reshard_after_forward (bool): Whether to reshard parameters after
                forward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                state._auto_reshard_after_forward = False
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward, fsdp_param_group.mesh_info
                        )
                    )

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward

    def set_modules_to_forward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: list[FSDPModule]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_all_reduce_hook(
        self,
        hook: Callable[[torch.Tensor], None],
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        Args:
            hook (Callable[[torch.Tensor], None]): User-defined all-reduce hook
                with expected signature ``hook(reduce_output: torch.Tensor) -> None``
                where ``reduce_output`` is the reduce-scatter output if only
                using FSDP or the all-reduce output if using native HSDP.
            stream (Optional[torch.cuda.Stream]): Stream to run the all-reduce
                hook in. This should only be set if not using native HSDP. If
                using native HSDP, the hook will run in the internally defined
                all-reduce stream used by the native HSDP all-reduce.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group._all_reduce_hook = hook
            if stream is not None:
                if fsdp_param_group._is_hsdp:
                    raise ValueError("stream cannot be set when using native HSDP")
                fsdp_param_group._all_reduce_hook_stream = stream

    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    @deprecated("Use `set_gradient_divide_factor` instead")
    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        """Use :py:meth:`set_gradient_divide_factor` instead"""
        self.set_gradient_divide_factor(factor)

    def set_gradient_divide_factor(self, factor: float) -> None:
        """
        Sets a custom divide factor for the gradient reduction. This might use
        a custom reduce op using NCCL's PreMulSum, which allows multiplying by
        the factor before reduction.

        Args:
            factor (float): Custom divide factor.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        """
        Sets whether to require the low-level collective communication
        primitives to exclusively use "sum"-type reductions, even if it comes
        at the cost of separate additional pre- or post-scaling operations.
        This is needed for example because NCCL currently supports zero-copy
        transfers only for this kind of collectives.

        NB: for MTIA devices, this is always implicitly enabled.

        NB: if `set_all_reduce_hook` is used under FSDP setup, the caller needs
        to ensure the custom all-reduce across FSDP units follow this strategy
        as well, as FSDP can no longer automatically handle that.

        Args:
            enable (bool): Whether to only ever use ReduceOp.SUM for comms.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.force_sum_reduction_for_comms = enable

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the FSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this FSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.unshard_in_backward = unshard_in_backward

    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        """
        Sets whether the temporary staging buffers used to send and receive data
        over collective communications should be allocated using the custom
        optimized allocator provided by the ProcessGroup itself (if any). This
        might allow the ProcessGroup to be more efficient. For example, when
        using NCCL, this enables it to leverage zero-copy transfers over SHARP
        (for NVLink and/or InfiniBand).

        Args:
            enable (bool): Whether to turn on ProcessGroup allocation.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.allocate_memory_from_process_group = enable

    def _set_unshard_async_op(self, async_op: bool):
        """
        Sets whether to use ``async_op=True`` or ``False`` for the pre-forward
        and pre-backward unshard op. This defaults to ``False`` but can be set
        to ``True`` with this method.

        Setting this to ``True`` allows the all-gather allocations to happen in
        the default stream, avoiding inter-stream memory fragmentation.
        However, you must use explicit prefetching (e.g. via :meth:`unshard`)
        in forward to still get overlap, and the pre-all-gather ops like dtype
        casting and copy-in will not overlap with compute.
        """
        self_module = cast(nn.Module, self)
        for module in self_module.modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.unshard_async_op = async_op

    def _get_fsdp_state(self) -> FSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No FSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        if not (fsdp_param_group := state._fsdp_param_group):
            return ret
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        with torch.no_grad():
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.reset_sharded_param()
        return ret


class UnshardHandle:
    """
    A handle to wait on a :meth:`FSDPModule.unshard` op.
    """

    def wait(self) -> None:
        """
        Waits on the unshard op. This ensures that the current stream can use
        the unsharded parameters, which are now registered to the module.
        """
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_group: Optional[FSDPParamGroup]):
        self._fsdp_param_group = fsdp_param_group

    def wait(self):
        if self._fsdp_param_group is not None:
            self._fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_group = None


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Registers a method on ``module`` to be considered a forward method for
    FSDP.

    FSDP all-gathers parameters pre-forward and optionally frees parameters
    post-forward (depending on ``reshard_after_forward``). FSDP only knows to
    do this for :meth:`nn.Module.forward` by default. This function patches a
    user-specified method to run the pre/post-forward hooks before/after the
    method, respectively. If ``module`` is not an :class:`FSDPModule`, then
    this is a no-op.

    Args:
        module (nn.Module): Module to register the forward method on.
        method_name (str): Name of the forward method.
    """
    if not isinstance(module, FSDPModule):
        # Make no-op to allow including both when using/not using FSDP
        return
    if not hasattr(module, method_name):
        raise ValueError(f"{type(module)} does not have a method {method_name}")
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        fsdp_state = self._get_fsdp_state()
        args, kwargs = fsdp_state._pre_forward(self, args, kwargs)
        out = orig_method(*args, **kwargs)
        return fsdp_state._post_forward(self, args, out)

    # Use `__get__` to make `wrapped_method` an instance method
    setattr(
        module,
        method_name,
        wrapped_method.__get__(module, type(module)),  # type:ignore[attr-defined]
    )


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"Expects FSDPModule but got {type(module)}: {module}")
