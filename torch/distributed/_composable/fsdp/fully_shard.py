from typing import Any, cast, Optional, Union

import typing_extensions

import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed._tensor import DeviceMesh

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


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
):
    """
    Shard module parameters across data parallel workers.

    This function applies fully sharded data parallelism (FSDP) or a variant to
    ``module``, a technique for memory savings at the cost of communication.
    Parameters are sharded across ``mesh``, and in turn, so are their gradients
    and optimizer states.

    The sharded parameters are all-gathered to construct the unsharded
    parameters for forward or backward computation. The unsharded parameters
    are freed after computation to save memory. The gradients are reduced
    across the mesh and divided by the mesh size for data parallelism. The
    optimizer step runs on the sharded parameters.

    Each call to ``fully_shard`` constructs one communication group that
    includes the parameters in ``module.parameters()`` except those already
    assigned to a group from a nested call. Each group's parameters and its
    gradients are communicated together in one collective, respectively.
    Constructing multiple groups across the model (e.g. "layer by layer")
    allows for peak memory savings and communication/computation overlap.

    Implementation-wise, the sharded parameters are represented as
    :class:`DTensor` s, sharded on dim-0, and the unsharded parameters are
    represented as :class:`Tensor` s. A module forward pre-hook all-gathers the
    parameters, and a module forward hook frees them. Similar backward hooks
    gather parameters and later free parameters/reduce gradients.

    Args:
        mesh (Optional[DeviceMesh]): This data parallel mesh defines the
            sharding and device. If 1D, then parameters are fully sharded
            across the 1D mesh (FSDP). If 2D, then parameters are sharded
            across the 0th dim and replicated across the 1st dim (HSDP). The
            mesh's device type gives the device type used for communication;
            if a CUDA or CUDA-like device type, then we use the current device.
        reshard_after_forward (Union[bool, int]): This controls the parameter
            behavior after forward and can trade off memory and communication:
            - If ``True``, then this reshards parameters after forward and
            all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
            after forward and avoids the all-gather in backward.
            - If an ``int``, then this represents the world size to reshard to
            after forward. It should be a non-trivial divisor of the ``mesh``
            shard dim size (i.e. excluding 1 and the dim size itself). A choice
            may be the intra-node size (e.g. ``torch.cuda.device_count()``).
            This allows the all-gather in backward to be over a smaller world
            size at the cost of higher memory usage than setting to ``True``.
            - The root FSDP state has its value specially set to ``False`` as a
            heuristic since its parameters would typically be immediately
            all-gathered for backward.
            - After forward, the parameters registered to the module depend on
            to this: The registered parameters are the sharded parameters if
            ``True``; unsharded parameters if ``False``; and the paramters
            resharded to the smaller mesh otherwise. To modify the parameters
            between forward and backward, the registered parameters must be the
            sharded parameters. For ``False`` or an ``int``, this can be done
            by manually resharding via :meth:`reshard`.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
    """
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
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    state = fully_shard.state(module)
    state.init(module, device, mp_policy)

    managed_modules = _get_managed_modules(module)
    params, buffers = _get_managed_states(managed_modules)
    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            module,
            mesh_info,
            post_forward_mesh_info,
            device,
            mp_policy,
            offload_policy,
        )

    # for dynamo
    for module in managed_modules:
        module._is_fsdp_managed_module = True  # type: ignore[assignment]
        module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"FSDP{cls.__name__}", (FSDP, cls), dct)
    module.__class__ = new_cls
    return module


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> typing_extensions.Never:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


class FSDP:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `FSDP` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, registering the sharded parameters
        to the module and freeing the unsharded parameters if needed. This
        method is *not* recursive.
        """
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> Optional["UnshardHandle"]:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then the user does not have to call
            :meth:`wait` on the returned handle if waiting on the unshard op
            in the module's pre-forward is tolerable. FSDP will wait on the
            pending unshard op in the pre-forward automatically.
        """
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard(async_op=async_op)
        handle = UnshardHandle(fsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one, meaning that FSDP
        should wait for gradient reduction to finish and clear internal data
        structures used for explicit prefetching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication. For HSDP, this controls
        both reduce-scatter and all-reduce together.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDP):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        # TODO: post_reduce_output += fsdp_param.sharded_param.grad
        # after reduce-scatter and before all-reduce
        raise NotImplementedError("requires_all_reduce is not yet supported in HSDP")
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDP):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDP):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward

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
    A handle to wait on the unshard op.

    Args:
        fsdp_param_group (FSDPParamGroup, optional): FSDP parameter group to
            unshard. This should be ``None`` iff the FSDP module does not
            manage any parameters, meaning the unshard is a no-op.
    """

    def __init__(self, fsdp_param_group: Optional[FSDPParamGroup]):
        self._fsdp_param_group = fsdp_param_group

    def wait(self):
        """
        Waits on the unshard op.

        This ensures that the current stream can use the unsharded parameters,
        which are now registered to the module.
        """
        if self._fsdp_param_group is not None:
            self._fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_group = None
