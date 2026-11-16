# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast, Literal, NoReturn, overload, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.nn as nn
from torch.distributed._composable import contract

from .._flex_shard import module_ops
from .._flex_shard.module_state import share_context
from . import _fsdp_engine
from ._fsdp_api import (
    AllGather,
    AllReduce,
    DataParallelMeshDims,
    GradientReductionHandle,
    MixedPrecisionPolicy,
    OffloadPolicy,
    ReduceScatter,
    UnshardHandle,
)
from ._fsdp_common import _dynamo_disable, FSDPMeshInfo, ShardPlacementFnResult
from ._fsdp_init import (
    _apply_to_module,
    _get_device_from_mesh,
    _get_mesh_info,
    _get_modules_and_states,
    _get_post_forward_mesh_info,
    _init_default_mesh,
    _init_param_group,
    _validate_mesh,
    _validate_module,
)
from ._fsdp_state import _get_module_fsdp_state, FSDPState


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torch.distributed.tensor import DeviceMesh

    from .._flex_shard.all_gather_layout import AllGatherLayout
    from .._flex_shard.bucket_runtime import BucketRuntime
    from .._flex_shard.bucket_storage import ShardedBucketStorage

__all__ = [
    "fully_shard",
    "FSDPModule",
    "GradientReductionHandle",
    "UnshardHandle",
    "register_fsdp_forward_method",
    "get_cls_to_fsdp_cls",
    "disable_fsdp_module_new_init",
    "share_comm_ctx",
]


cls_to_fsdp_cls: dict[type, type] = {}


def get_cls_to_fsdp_cls() -> dict[type, type]:
    return cls_to_fsdp_cls


@overload
# pyrefly: ignore [inconsistent-overload]
def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int | None = ...,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
) -> FSDPModule: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int | None = ...,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    dp_mesh_dims: DataParallelMeshDims | None = ...,
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
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
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

    Parameter dtypes and ``grad_dtype`` may change (e.g. ``module.bfloat16()``)
    until lazy initialization at the first forward or
    :meth:`FSDPModule.unshard`; later changes are unsupported. An explicit
    ``grad_dtype`` survives dtype conversions and is inherited by a parameter
    that ``load_state_dict(assign=True)`` registers without one. Device moves
    preserve sharded gradient dtypes and leave pending gradients on their
    original device.

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

    When called with a list (``fully_shard([a, b, ...])``), the model's
    forward may run only a subset of the grouped modules, with the rest
    called later in the same iteration. Chunked-loss training with
    ``fully_shard([norm, head])`` is the motivating case: the main forward
    runs norm only, then head is invoked per chunk. Caveats:

    - Each standalone per-chunk invocation registers its own post_backward
      autograd node, so N chunk calls produce N reduce-scatters for that
      group.
    - ``mp_policy.cast_forward_inputs`` and ``mp_policy.output_dtype``
      both apply per module in the group — every invocation (including
      each standalone per-chunk call) casts its inputs to ``param_dtype``
      and its output to ``output_dtype``.

    .. note::
       If ``forward()`` or ``backward()`` raises, FSDP's per-iteration
       state (iteration forward-root marker, grouped-module run
       trackers, in-flight collective state, per-group training states)
       is left in an undefined condition. To recover and run another
       iteration, call :meth:`FSDPModule.reset_iter_state` on the root
       FSDP module. The failed iteration's gradients are discarded,
       including any ``no_sync`` / HSDP partial-reduce accumulation
       state.

    Args:
        module (Union[nn.Module, List[nn.Module]]): The module or modules to
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
              resharded to the smaller mesh otherwise. For ``False`` or an
              ``int``, the sharded parameters can be registered by manually
              resharding via :meth:`reshard`.
            - Modifying the parameters between forward and backward is not
              supported: the backward all-gather is not ordered after such
              writes, so backward may see the old values.
        shard_placement_fn (Optional[Callable[[nn.Parameter], Optional[Shard | ShardPlacementResult]]]):
            This callable can be used to override the sharding placement and/or
            mesh for a parameter. It can return:

            - ``None``: Use default sharding (Shard(0)) on the mesh passed to
              ``fully_shard``.
            - :class:`Shard`: Shard the parameter on the specified dimension
              using the mesh passed to ``fully_shard``.
            - ``ShardPlacementResult``: Specify both the shard placement
              (``placement``) and a custom ``FSDPMeshInfo`` (``mesh_info``).
              Both are currently private types. This allows different
              parameters to be sharded across different process groups, enabling
              use cases like Mixture of Experts where expert params use a
              different mesh than regular params.

            Any dim may be sharded unevenly, as ``torch.chunk`` splits it, so later
            ranks may hold fewer or no rows. On a nonzero dim, all-gather
            extensions of such parameters need the 5-argument
            ``fsdp_pre_all_gather``, must pad their inputs on those ranks, and
            must drop the padding in ``fsdp_post_all_gather`` (see
            :class:`~torch.distributed.fsdp.experimental.AllGatherInput`).
            Inputs left unpadded raise on those ranks only, while the others
            wait in the all-gather.
            :func:`~torch.distributed.fsdp.experimental.reduce_scatter_input_fn_with_native_copy`
            does not support such parameters.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
        ignored_params: Optional(Set[nn.Parameter]): The set of parameters to be
            ignored by FSDP. They will not be sharded, nor moved to the device
            during init, nor have their gradients reduced in backward.
        dp_mesh_dims (Optional[DataParallelMeshDims]): When provided,
            ``mesh`` is treated as the full SPMD mesh, and parameters should be
            DTensors on this mesh with ``Replicate()`` on all DP dimensions.
            The ``shard`` field names which dim(s) FSDP shards on (multiple
            dims are flattened). The ``replicate`` field names the HSDP
            replication dim(s) (multiple dims are flattened).

    Returns:
        FSDPModule: The module with FSDP applied (in-place).
    """
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")
    _validate_module(module, "fully_shard")
    mesh = mesh or _init_default_mesh()
    _validate_mesh(mesh, dp_mesh_dims)
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
    device = _get_device_from_mesh(mesh)
    auto_reshard_after_forward = reshard_after_forward is None
    # If the user does not provide ``reshard_after_forward``, we set it to True.
    # During lazy_init, we identify which module is the root and override its value to False
    if isinstance(mesh_info, FSDPMeshInfo):
        if (
            mesh_info.is_spmd_mesh
            and not isinstance(reshard_after_forward, bool)
            and isinstance(reshard_after_forward, int)
        ):
            raise NotImplementedError(
                "reshard_after_forward as int is not yet supported with "
                "SPMD mesh (dp_mesh_dims)"
            )
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward if not auto_reshard_after_forward else True,  # type: ignore[arg-type]
            mesh_info,
        )
    else:
        # DDPMeshInfo: no sharding, so no post-forward resharding needed
        post_forward_mesh_info = None
    arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
        module, device, ignored_params
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    _init_param_group(
        state,
        params,
        modules,
        mesh_info,
        post_forward_mesh_info,
        device,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        reshard_after_forward=reshard_after_forward
        if not auto_reshard_after_forward
        else True,
    )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    _apply_to_module(
        modules, cls_to_fsdp_cls, FSDPModule, "FSDP", _unimplemented_deepcopy
    )
    return arg_module


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


_fsdp_module_new_init_disabled: ContextVar[bool] = ContextVar(
    "_fsdp_module_new_init_disabled", default=False
)


@contextmanager
def disable_fsdp_module_new_init() -> Iterator[None]:
    """Make ``FSDPModule.__new__`` return an uninitialized instance of the
    wrapper class itself, for rebuilding a module from saved state.

    Do not call an FSDP class inside this context: Python would run the original
    ``__init__`` on that instance, returning a wrapper with no FSDP state.
    """
    token = _fsdp_module_new_init_disabled.set(True)
    try:
        yield
    finally:
        _fsdp_module_new_init_disabled.reset(token)


class FSDPModule:
    # Index in MRO where the original class is found.
    # For FSDP: [FSDP<Orig>, FSDPModule, Orig, ...] -> index 2
    # Subclasses like ReplicateModule override this.
    _orig_cls_mro_index: int = 2

    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        Under ``disable_fsdp_module_new_init``, return an uninitialized
        instance of ``cls`` itself, so a module rebuilt from guard state keeps
        the wrapper type its guards match.
        """
        orig_cls = cls.__mro__[cls._orig_cls_mro_index]
        if _fsdp_module_new_init_disabled.get():
            # pyrefly: ignore [no-matching-overload]
            return orig_cls.__new__(cls)
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters and registers the sharded parameters
        to the module, also while its forward is in progress (see
        :meth:`complete_grouped_forward`). This method is *not* recursive. By
        default, this frees the unsharded parameter storage; call
        :meth:`set_keep_unsharded_storage` to preserve it.
        """
        module_ops.reshard(_fsdp_runtimes(self))

    def complete_grouped_forward(self, output: Any = None) -> Any:
        """
        Completes the forward of the parameter group that ``fully_shard``
        made of this module and the modules passed with it, if only some of
        them ran. FSDP completes a group's forward when its last module
        finishes, or at the end of the root module's forward. A group whose
        modules run on their own after that, e.g. the output projection of
        ``fully_shard([norm, output])`` called by a chunked loss, stays in its
        forward until this method completes it: the parameters reshard as
        :meth:`set_reshard_after_forward` says, and the backward hooks are
        registered on the tensors in ``output``, which is returned. Call it
        after each such call, or once with no ``output`` to end a forward-only
        pass, e.g. before capturing a graph. This method is a no-op if the
        group's forward is not in progress, and is *not* recursive.

        Args:
            output (Any): The output of the module's call, whose tensors get
                the backward hooks.

        Returns:
            Any: ``output``, as autograd needs it for the hooks.
        """
        return self._get_fsdp_state()._complete_grouped_forward(output)

    @contextmanager
    def partial_backward(self, *, outer_backward_follows: bool) -> Iterator[None]:
        """
        Runs forwards and backwards through this module alone, several times
        within one step, e.g. the output projection of a chunked loss, which
        computes the loss chunk by chunk from the detached output of the rest
        of the model and backpropagates each chunk through this module. Call
        it after the root module's forward. Within the scope:

        - The module's parameters are all-gathered once, on entry, and stay
          unsharded.
        - Each call of the module completes its parameter group's forward (see
          :meth:`complete_grouped_forward`).
        - Its gradients accumulate without being reduced.
        - Backwards leave the other FSDP modules alone: their parameters,
          gradients awaiting reduction and backward prefetch order stay for the
          backward through them.

        On exit, if a backward ran within the scope, the module's gradients are
        reduced and its parameters resharded as its settings say
        (:meth:`set_requires_gradient_sync`, :meth:`set_requires_all_reduce`,
        :meth:`set_reshard_after_backward`), except:

        - With ``outer_backward_follows`` and a parameter group that the module
          shares with others (``fully_shard([norm, output])``), the outer
          backward does both, so that the group's gradients are reduced
          together.
        - With :meth:`set_manual_backward_finalization`, :meth:`finalize_backward`
          does the reduction and leaves the residency to the caller.

        Without ``outer_backward_follows``, exiting then ends the backward as the
        end of a backward does, e.g. waiting for the reductions on the last
        backward (:meth:`set_is_last_backward`). Exiting after a forward-only use
        reshards the parameters. Exiting on an exception reduces nothing; call
        :meth:`reset_iter_state` on the root module to recover. The settings
        themselves do not change.

        Args:
            outer_backward_follows (bool): Whether a backward through the rest
                of the model, e.g. from the gradients that a chunked loss
                accumulated for the inputs of this module, follows the scope
                in the same step.
        """
        with self._get_fsdp_state()._partial_backward(outer_backward_follows):
            yield

    def get_reshard_after_forward(self) -> bool | int:
        """
        Returns whether this module reshards its parameters after forward, or
        the size of the group it reshards them to (see
        :meth:`set_reshard_after_forward` and ``fully_shard``'s
        ``reshard_after_forward``). The root module reports ``False`` for
        ``fully_shard``'s default once its first forward made it the root.
        """
        return module_ops.get_reshard_after_forward(_first_storage(self))

    def get_reshard_after_backward(self) -> bool:
        """
        Returns whether this module reshards its parameters after backward
        (see :meth:`set_reshard_after_backward`).
        """
        storage = _first_storage(self)
        return storage._reshard_after_backward if storage is not None else True

    def get_requires_gradient_sync(self) -> bool:
        """
        Returns whether this module reduces its gradients (see
        :meth:`set_requires_gradient_sync`).
        """
        storage = _first_storage(self)
        return storage._requires_gradient_sync if storage is not None else True

    def get_requires_all_reduce(self) -> bool:
        """
        Returns whether this module all-reduces its gradients across replicas
        with HSDP (see :meth:`set_requires_all_reduce`), which
        :meth:`set_requires_gradient_sync` also sets.
        """
        storage = _first_storage(self)
        return storage._requires_all_reduce if storage is not None else True

    def get_is_last_backward(self) -> bool:
        """
        Returns whether the next backward is the last one (see
        :meth:`set_is_last_backward`).
        """
        return self._get_fsdp_state()._state_ctx.is_last_backward

    def set_keep_unsharded_storage(self, keep: bool, *, recurse: bool = True) -> None:
        """
        Sets whether to keep the unsharded parameter storage after resharding.
        Keeping the storage preserves its data pointers across reshard and
        unshard, which is useful for CUDA graph capture, at the cost of higher
        memory usage. By default, FSDP frees the unsharded parameter storage
        when resharding.

        Args:
            keep (bool): Whether to keep the unsharded parameter storage after
                resharding.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        module_ops.set_keep_unsharded_storage(_fsdp_runtimes(self, recurse), keep)

    def unshard(self, async_op: bool = False) -> UnshardHandle | None:
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
        handle = module_ops.ModuleUnshardHandle(
            self._get_fsdp_state()._unshard(async_op)
        )
        if async_op:
            return handle
        handle.wait()
        return None

    @_dynamo_disable
    def reset_iter_state(self) -> None:
        """
        Resets FSDP's per-iteration state after an exception aborted a
        forward or backward mid-flight. The supported recovery workflow is:

        1. Catch the exception from ``forward()`` or ``backward()``.
        2. Call ``reset_iter_state()`` on the *root* FSDP module.
        3. Run the next iteration normally.

        The reset waits on any in-flight all-gather/reduce-scatter events,
        reshards every parameter group, and clears iteration trackers
        (``iter_forward_root``, grouped-module forward passes, pending
        prefetches, per-group training states). Any in-flight gradient
        reductions are discarded: the failed iteration's gradients are
        lost, including HSDP partial-reduce-accumulation state and
        ``no_sync`` grad-accumulation state. Callers doing gradient
        accumulation should treat the microbatch sequence as invalidated
        and restart it.

        Reset invalidates an outstanding :class:`GradientReductionHandle`
        that owns this module's shared communication context. Calling
        :meth:`GradientReductionHandle.wait` on that handle raises
        ``RuntimeError``. Reset from a different FSDP root sharing that context
        raises until the owning root waits or resets.

        Must be called on the root FSDP module — i.e. the module the
        top-level ``fully_shard`` was applied to, equivalently the
        module first forwarded. Calling on a non-root module raises
        ``RuntimeError``.
        """
        self._get_fsdp_state()._reset_iter_state()

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_manual_backward_finalization(self, enabled: bool) -> None:
        """
        Set whether the caller must finalize backward.

        This must be called on the root FSDP module. When enabled, manual
        finalization supersedes :meth:`set_is_last_backward`. Call
        :meth:`finalize_backward` after all backward passes in the logical
        backward operation and before reading or clearing gradients. Otherwise,
        gradients may remain unreduced and backward iteration state is retained.
        Gradient synchronization and parameter resharding follow their current
        settings.

        Set this before backward. The mode cannot change after backward starts
        until the backward iteration is finalized or reset.
        """
        self._get_fsdp_state()._set_manual_backward_finalization(enabled)

    @overload
    def finalize_backward(self, *, async_op: Literal[False] = False) -> None: ...

    @overload
    def finalize_backward(
        self, *, async_op: Literal[True]
    ) -> GradientReductionHandle: ...

    @overload
    def finalize_backward(
        self, *, async_op: bool
    ) -> GradientReductionHandle | None: ...

    @_dynamo_disable
    def finalize_backward(
        self, *, async_op: bool = False
    ) -> GradientReductionHandle | None:
        r"""
        Finalize backward on the calling thread.

        Enable manual finalization before forward, then call this after all
        backward passes in the logical backward operation. This completes
        pending gradient reduction and resharding according to their current
        settings. Calling this before the root module's first forward is a no-op.
        It is also safe after a completed backward that did not reach any
        FSDP-managed parameters. Do not call it between forward and backward.

        If several backward passes precede one finalization, disable gradient
        synchronization for those backward passes and re-enable it before
        finalization.

        Manual finalization lets callers choose a finalization point that is
        separate from any backward call. This supports schedules that represent
        gradient reduction as a separate action. CUDA graph capture does not
        support CPU gradient offload or an outstanding asynchronous unshard.

        Partial gradient reduction for ``replicate()`` and HSDP is not
        supported. Enable all-reduce before finalization.

        Finalization follows these call paths on the FlexShard engine::

            automatic final backward
              -> BucketCommContext end-of-backward callback
                -> finish the buckets backward left, wait for the reductions

            FSDPModule.finalize_backward(async_op=False)
              -> BucketCommContext.start_finalize_backward()
              -> wait for the reductions, end the backward iteration

            FSDPModule.finalize_backward(async_op=True)
              -> BucketCommContext.start_finalize_backward()
              -> GradientReductionHandle.wait()
                -> wait for the reductions, end the backward iteration

        Args:
            async_op (bool): If ``True``, return a
                :class:`GradientReductionHandle` without waiting for gradient
                reduction. The caller must call :meth:`wait` before using the
                gradients or starting more work on this FSDP module or another
                FSDP root that shares its communication context. If ``False``,
                wait before returning.
        """
        return self._get_fsdp_state()._finalize_backward(async_op)

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together. This is the equivalence of
        `no_sync` in FSDP1.

        Unsynchronized gradients accumulate on unsharded parameters in
        ``MixedPrecisionPolicy.reduce_dtype`` if set, and otherwise as the
        parameter's ``grad_dtype`` specifies (the original dtype if unset). Sharded gradients and HSDP
        buffers awaiting all-reduce remain separate. CPU offload moves only
        fully reduced sharded gradients to CPU.

        ``zero_grad()`` clears only the currently registered parameters' gradients,
        leaving other parameter copies and pending all-reduce buffers intact.

        Before gradient clipping or an optimizer step, enable the required
        reductions and call ``set_is_last_backward(True)`` for the final backward.
        Reshard before updating parameters.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        module_ops.set_requires_gradient_sync(
            _fsdp_storages(self, recurse), requires_gradient_sync
        )

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        module_ops.set_requires_all_reduce(
            _fsdp_storages(self, recurse), requires_all_reduce
        )

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
                _fsdp_engine.set_reshard_after_forward(state, reshard_after_forward)

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Call :meth:`reshard` on each FSDP module on every rank before updating
        its sharded parameters.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        module_ops.set_reshard_after_backward(
            _fsdp_storages(self, recurse), reshard_after_backward
        )

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
        states = [module._get_fsdp_state() for module in modules]
        _fsdp_engine.set_modules_to_prefetch(
            self._get_fsdp_state(), states, forward=True
        )

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
        states = [module._get_fsdp_state() for module in modules]
        _fsdp_engine.set_modules_to_prefetch(
            self._get_fsdp_state(), states, forward=False
        )

    def set_custom_all_gather(self, comm: AllGather) -> None:
        """
        Overrides the default ``all_gather`` communication behavior,
        to have better control over the communication and memory usage.
        See `Comm` and `ReduceScatter` for details.

        This only works with the default all-gather layout,
        :class:`~torch.distributed.fsdp.experimental.DefaultAllGatherLayout`,
        which packs the comm's input into the output that the comm allocated; a
        custom layout carries its own comm instead (see
        :meth:`set_all_gather_layout`). The comm can be replaced at any time,
        since it owns no outputs.

        Args:
            comm (AllGather): Custom all-gather communication.
        """
        module_ops.set_custom_all_gather(_fsdp_runtimes(self), comm)

    def set_custom_reduce_scatter(self, comm: ReduceScatter) -> None:
        """
        Overrides the default ``reduce_scatter`` communication behavior,
        to have better control over the communication and memory usage.
        See `Comm` and `ReduceScatter` for details.

        Args:
            comm (ReduceScatter): Custom reduce_scatter communication.
        """
        module_ops.set_custom_reduce_scatter(_fsdp_storages(self), comm)

    def set_custom_all_reduce(self, comm: AllReduce) -> None:
        """Override HSDP or ``replicate_with_fsdp`` all-reduce communication.

        See ``AllReduce`` for the in-place reduction and stream contract. The
        primitive runs on FSDP's internal all-reduce stream, which is distinct
        from the compute stream. FSDP handles gradient scaling, accumulation,
        and buffer lifetime, and runs the optional all-reduce hook afterward.
        ``set_requires_all_reduce(False)`` suppresses this collective.

        This applies to this module's parameter group, not child FSDP modules.

        .. warning::
            This API is experimental and may change without backward
            compatibility.

        Args:
            comm (Optional[AllReduce]): Custom all-reduce communication.
                ``None`` restores ``DefaultAllReduce``.
        """
        module_ops.set_custom_all_reduce(_fsdp_runtimes(self), comm)

    def set_all_reduce_hook(
        self,
        hook: Callable[[torch.Tensor], None],
        *,
        stream: torch.cuda.Stream | None = None,
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

        A module with several parameter groups (meshes from ``shard_placement_fn``)
        runs the hook on each group's reduction output.
        """
        module_ops.set_all_reduce_hook(_fsdp_storages(self), hook, stream)

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
        module_ops.set_gradient_divide_factor(_fsdp_storages(self), factor)

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
        module_ops.set_force_sum_reduction_for_comms(_fsdp_storages(self), enable)

    def set_all_gather_layout(
        self, layout: AllGatherLayout, /, *, recurse: bool = True
    ) -> None:
        r"""Set the layout that packs and finalizes a parameter group's all-gather.

        .. warning::
            This API is experimental. The layout contract and supported FSDP
            internals may change without backward compatibility.

        The layout packs the group's all-gather inputs before the collective and
        turns the collective output into the parameters' all-gather outputs
        after it completes. This is the only way to install a layout.
        :class:`~torch.distributed.fsdp.experimental.DefaultAllGatherLayout`
        packs rank-major input into the collective output and copies the output
        with its ``output_fn``, e.g.
        ``DefaultAllGatherLayout(all_gather_output_fn_with_native_copy)`` for the
        native copy. It is the only layout that works with the comms that
        :meth:`set_custom_all_gather`, :meth:`set_symm_mem_for_comm` and
        :meth:`set_allocate_memory_from_process_group_for_comm` install, which
        expect that copy-in. A custom layout, e.g. one whose collective writes
        parameter-contiguous output, packs its own input and carries its own
        all-gather as its ``comm``, which FSDP passes each call's layout
        metadata; without a ``comm``, it runs FSDP's default all-gather. A
        layout that keeps no per-call state can serve any number of modules. Set
        custom layouts before the first unshard: layouts change only between
        iterations, while no all-gather is pending and parameters are resharded,
        and never after parameters adopt outputs from a custom layout, since
        only the layout that created those outputs can refill them. FSDP checks
        every group before changing any.

        Args:
            layout (AllGatherLayout): Layout to install; pass
                ``DefaultAllGatherLayout()`` to restore the default.
            recurse (bool): Whether to also set the layout for all nested FSDP
                modules. Defaults to ``True``.
        """
        module_ops.set_all_gather_layout(_fsdp_runtimes(self, recurse), layout)

    def set_reduce_scatter_input_fn(
        self, fn: Callable | None, /, *, recurse: bool = True
    ) -> None:
        r"""Set the function that prepares reduce-scatter inputs.

        .. warning::
            This API is experimental. The callback signature and supported FSDP
            internals may change without backward compatibility.

        ``copy_in = fn(unsharded_grads, shard_dims=shard_dims,
        world_size=world_size)`` runs before FSDP allocates the reduce-scatter
        input and may replace entries of ``unsharded_grads``, e.g. with
        reordered copies.
        ``copy_in(reduce_scatter_input)`` then fills that flat buffer so that,
        viewed as ``(world_size, -1)``, row ``r`` holds each gradient's padded
        shard ``r`` in order, cast from any gradient dtype to the buffer's. Both
        run under ``torch.no_grad()`` on the current stream, also when
        ``world_size`` is 1. FSDP frees ``copy_in`` and the gradients afterward,
        so keep neither. See :mod:`torch.distributed.fsdp.experimental` for a
        native implementation.

        Args:
            fn (Optional[Callable]): Function returning the copy-in function, or
                ``None`` to restore the default.
            recurse (bool): Whether to also set the function for all nested FSDP
                modules. Defaults to ``True``.
        """
        module_ops.set_reduce_scatter_input_fn(_fsdp_storages(self, recurse), fn)

    def set_reduce_scatter_unused_params(
        self, reduce_scatter_unused_params: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets whether to include zero gradients for parameters that did not
        receive a gradient in backward. This is needed when different ranks
        use different parameters due to conditional control flow (e.g.
        multi-modal models, mixture of experts), causing mismatched
        reduce-scatter collectives. Similar to DDP's
        ``find_unused_parameters``.

        Parameters requiring gradients with explicit ``grad_dtype=None``
        require a non-``None`` ``reduce_dtype`` so zero gradients have the
        same dtype on every rank; otherwise, backward raises an error.

        Args:
            reduce_scatter_unused_params (bool): Whether to include zero
                gradients for unused parameters in gradient reduction.
            recurse (bool): Whether to set for all FSDP submodules or just
                the passed-in module.
        """
        module_ops.set_reduce_scatter_unused_params(
            _fsdp_storages(self, recurse), reduce_scatter_unused_params
        )

    def set_reduce_scatter_max_input_buffers(
        self, max_input_buffers: int, *, recurse: bool = True
    ) -> None:
        """
        Sets how many gradient reduce-scatter input buffers may be in flight at
        once -- the copy-in (``chunk_cat``) buffer cap-K (experimental).

        FSDP keeps **1** such buffer in flight by default, so the compute stream
        must wait on the previous reduce-scatter before the next copy-in can
        reuse that buffer. When the reduce-scatter is exposed (communication
        slower than the backward compute meant to hide it), that recycle wait
        stalls the compute stream. Raising the cap lets the next copy-in write a
        **fresh** buffer instead of waiting -- removing the stall -- at the cost
        of extra peak memory for the retained buffers. The copy-in stays on the
        compute stream; there is no extra stream and no ``record_stream``. This
        helps only when the reduce-scatter is exposed.

        Args:
            max_input_buffers (int): Max reduce-scatter input buffers retained in
                flight (the memory<->overlap dial); must be ``>= 1``. ``1`` is
                FSDP's default behavior (single buffer; the exposed-RS stall). A
                small value (e.g. ``2``) bounds peak memory and adds no stall as
                long as it is ``>=`` the reduce-scatter pipeline depth (otherwise
                an exposed reduce-scatter trades back a tail stall); a larger
                value retains more buffers for deeper overlap at higher peak
                memory.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                module._get_fsdp_state()._set_reduce_scatter_max_input_buffers(
                    max_input_buffers
                )

    def set_separate_reduce_scatter_group(
        self, enable: bool = True, *, recurse: bool = True
    ) -> None:
        """
        Enables (or disables) running gradient reduce-scatter on its own process
        group so it can overlap with all-gather in the backward pass
        (experimental).

        By default FSDP runs all-gather and reduce-scatter on separate CUDA
        streams but through the **same** process group -- one NCCL communicator,
        which processes one collective at a time and so serializes them on the
        wire. When enabled, FSDP creates a dedicated process group over the shard
        ranks (``dist.new_group(..., use_local_synchronization=True)``) -- one
        per distinct set of shard ranks, typically a single communicator -- so
        the two collectives can progress concurrently when the network can
        sustain it. This is collective for each shard rank set: like other FSDP
        comm setup, call it consistently across ranks using this FSDP mesh.

        Args:
            enable (bool): ``True`` (default) gives reduce-scatter its own
                process group; ``False`` resets it to the shared shard/all-gather
                group.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        # Groups are created per set of shard ranks, so meshes that shard over
        # the same ranks share one communicator (typically one total)
        module_ops.set_separate_reduce_scatter_group(
            _fsdp_storages(self, recurse), enable
        )

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the FSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this FSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
        module_ops.set_unshard_in_backward(_fsdp_storages(self), unshard_in_backward)

    def set_allocate_memory_from_process_group_for_comm(self, enable: bool) -> None:
        """
        Sets whether the temporary staging buffers used to send and receive data
        over collective communications should be allocated using the custom
        optimized allocator provided by the ProcessGroup itself (if any). This
        might allow the ProcessGroup to be more efficient. For example, when
        using NCCL, this enables it to leverage zero-copy transfers over SHARP
        (for NVLink and/or InfiniBand).

        This cannot be used together with :meth:`set_custom_all_gather` or
        :meth:`set_custom_reduce_scatter` as those APIs allow for
        finer-grained control over each communication, and this method cannot
        determine their staging buffer allocation strategy. ProcessGroup
        allocation only works with the default all-gather layout,
        :class:`~torch.distributed.fsdp.experimental.DefaultAllGatherLayout`.

        Args:
            enable (bool): Whether to turn on ProcessGroup allocation.
        """
        module_ops.set_allocate_memory_from_process_group(_fsdp_runtimes(self), enable)

    def set_symm_mem_for_comm(self, backend: Literal["NCCL"] = "NCCL") -> None:
        """
        Sets the symmetric memory (``symm_mem``) backend for allocating the
        staging buffers used in all-gather collectives. This allows NCCL to use
        optimized all-gather implementations via symmetric memory. Such
        optimization may depend on the topology of the system.  For single node,
        Copy Engine All-Gather may be used. For multi-node, Symmetric Kernel
        All-Gather may be used.

        To enable Copy Engine All-Gather, you need to set the NCCL process group
        with the zero-CTA policy.
        ```python
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
        ```
        Alternatively, you can set the environment variable `NCCL_CTA_POLICY` to 2.
        ```bash
        export NCCL_CTA_POLICY=2
        ```
        For more details, see [Copy Engine
        Collectives](https://docs.pytorch.org/docs/2.11/symmetric_memory.html#copy-engine-collectives).

        This cannot be used together with :meth:`set_custom_all_gather` or
        :meth:`set_custom_reduce_scatter`, and only works with the default
        all-gather layout,
        :class:`~torch.distributed.fsdp.experimental.DefaultAllGatherLayout`.

        Args:
            backend (str): The symmetric memory backend to use. Defaults to
                ``"NCCL"``. Currently, only ``"NCCL"`` is supported.
        """
        module_ops.set_symm_mem(_fsdp_runtimes(self), backend)

    def set_unshard_in_current_stream(
        self, unshard_in_current_stream: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets whether FSDP issues its pre-forward and pre-backward all-gathers
        on the current stream instead of on its separate all-gather streams.
        The current stream is the one that is current when FSDP issues the
        unshard, normally the one the module's forward and backward run on, so
        the all-gather buffers share that stream's memory. Defaults to
        ``False``.

        Setting this to ``True`` avoids inter-stream memory fragmentation: the
        CUDA caching allocator does not reuse memory freed on one stream for
        allocations on another. In return, the copy-in and dtype cast before
        each all-gather run on the current stream and do not overlap with
        compute.

        With this set, FSDP's own unshards run like your own :meth:`unshard`
        calls, on the current stream: the pre-forward and pre-backward unshard
        like ``unshard(async_op=False)``, and a prefetch like
        ``unshard(async_op=True)``.

        An all-gather issued on the current stream overlaps with compute only
        if it is issued ahead of time. In forward, configure explicit
        prefetching with :meth:`set_modules_to_forward_prefetch` or call
        :meth:`unshard` with ``async_op=True``; otherwise each module's
        all-gather is exposed. Backward prefetching needs no configuration.

        Args:
            unshard_in_current_stream (bool): Whether to issue the all-gathers
                on the current stream.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        module_ops.set_unshard_async_op(
            _fsdp_storages(self, recurse), unshard_in_current_stream
        )

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
        module_ops.set_unshard_async_op(_fsdp_storages(self, recurse=True), async_op)

    def _free_sharded_params(self) -> None:
        """
        Frees the storage of this module's sharded parameters, which inference
        on persistent unsharded parameters (:meth:`set_keep_unsharded_storage`)
        does not need. Call :meth:`_restore_sharded_params` before reading or
        writing the sharded parameters or unsharding again.
        """
        module_ops.free_sharded_params(_fsdp_runtimes(self))

    def _restore_sharded_params(self) -> None:
        """
        Re-allocates the storage that :meth:`_free_sharded_params` freed, with
        undefined values, e.g. to load new weights into the sharded parameters.
        """
        module_ops.restore_sharded_params(_fsdp_runtimes(self))

    def _get_fsdp_state(self) -> FSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No FSDP state found on {self}")
        return state

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Any:
        apply = super()._apply  # type: ignore[misc]
        return module_ops.apply_to_params(
            _fsdp_runtimes(self), lambda: apply(fn, recurse=recurse)
        )


def _fsdp_states(module: FSDPModule, recurse: bool) -> list[FSDPState]:
    self_module = cast(nn.Module, module)
    modules = self_module.modules() if recurse else [self_module]
    return [m._get_fsdp_state() for m in modules if isinstance(m, FSDPModule)]


def _fsdp_storages(
    module: FSDPModule, recurse: bool = False
) -> list[ShardedBucketStorage]:
    """The engine storages of ``module``'s groups, or of all FSDP modules'
    under it."""
    return [st for s in _fsdp_states(module, recurse) for st in s.storages]


def _fsdp_runtimes(module: FSDPModule, recurse: bool = False) -> list[BucketRuntime]:
    return [r for s in _fsdp_states(module, recurse) for r in s.runtimes]


def _first_storage(module: FSDPModule) -> ShardedBucketStorage | None:
    """The storage of the first group, which holds the settings that the
    setters give all of ``module``'s groups."""
    return next(iter(_fsdp_storages(module)), None)


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


def share_comm_ctx(modules: list[FSDPModule]) -> None:
    """
    Share cuda streams for multiple FSDPModules

    Example usage:
        from torch.distributed.fsdp import share_comm_ctx
        share_comm_ctx([fsdp_model_1, fsdp_model_2, ...])

    For Pipeline Parallelism (PP), each model chunk is a FSDP root. We want
    to share cuda streams for all-gather, reduce-scatter, and all-reduce.
    This avoids allocating inter-stream memory framgmentation

    Args:
        modules (List[FSDPModule]): modules to share cuda streams
    """
    if len(modules) == 0:
        return
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"Expects list of FSDPModules but got {module}")
    fsdp_states = [module._get_fsdp_state() for module in modules]
    share_context(fsdp_states)


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError(f"Expects FSDPModule but got {type(module)}: {module}")
