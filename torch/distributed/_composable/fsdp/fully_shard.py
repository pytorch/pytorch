from typing import Any, cast, List, Optional, Union

import torch
import torch.nn as nn

from torch.distributed._composable import contract
from torch.distributed._composable_state import _insert_module_state
from torch.distributed._tensor import DeviceMesh, DTensor

from ._fsdp_api import CommPolicy, InitPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import _normalize_device, FSDPMeshInfo, HSDPMeshInfo
from ._fsdp_init import (
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _materialize_meta_modules,
    _move_params_and_buffers_to_device,
)
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import _get_module_fsdp_state, FSDPState


@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    device: Union[torch.device, int, str] = "cuda",
    reshard_after_forward: Union[bool, int] = True,
    init_policy: InitPolicy = InitPolicy(),
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    comm_policy: CommPolicy = CommPolicy(),
):
    """
    Args:
        device (Union[torch.device, int, str]): This is the device used for
            both FSDP initialization and training. Managed parameters and
            buffers are on this device before this function returns. Aside from
            testing, this generally should be a CUDA or CUDA-like device. CUDA
            and CPU devices use NCCL and gloo for collective communications,
            respectively. (Default: ``"cuda"``)
        reshard_after_forward (Union[bool, int]): This controls the parameter
            behavior after forward and can trade off memory and communication.
            - If ``True``, then this reshards parameters after forward and
            all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
            after forward and avoids the all-gather in backward.
            - If an ``int``, then this represents the world size to reshard to
            after forward. It should be a number between 1 and the ``mesh``
            shard dimension size exclusive. A common choice may be the
            intra-node size (i.e. ``torch.cuda.device_count()``).

    .. note:: To use a specific CUDA device, you can either set the current
        device via :func:`torch.cuda.device` or :func:`torch.cuda.set_device`
        while passing ``"cuda"`` to ``device``, or you can pass a CUDA device
        with its index directly to ``device``.
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    device = _normalize_device(device)
    if (offload_type := offload_policy.offload_type) not in (None, "cpu"):
        raise NotImplementedError(f"Offloading only supports 'cpu', not {offload_type}")
    if device == torch.device("meta") and offload_type == "cpu":
        raise ValueError("device='meta' is not supported with CPU offloading")
    if (
        comm_policy.forward_prefetch_limit != 1
        or comm_policy.backward_prefetch_limit != 1
    ):
        raise NotImplementedError("Setting the prefetch policy is not supported yet")

    mesh = mesh or _init_default_fully_shard_mesh(device.type)
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    elif mesh.ndim == 2:
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    if device.type != mesh.device_type:
        raise ValueError(
            f"device and mesh must be of the same type but got {device.type} "
            f"for device and {mesh.device_type} for mesh"
        )
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )
    state = fully_shard.state(module)
    _insert_module_state(module, state)
    state._module = module
    state._mp_policy = mp_policy
    state._device = device
    state._pre_forward_hook_handle = state._module.register_forward_pre_hook(
        state._pre_forward, prepend=True, with_kwargs=True
    )
    state._post_forward_hook_handle = state._module.register_forward_hook(
        state._post_forward, prepend=False
    )

    managed_modules = _get_managed_modules(module)
    _materialize_meta_modules(
        managed_modules, init_policy.module_init_fn, state._device
    )
    named_params = {p: n for n, p in module.named_parameters()}
    params, buffers = _get_managed_states(managed_modules, named_params)
    _move_params_and_buffers_to_device(
        params, buffers, state._device, mesh_info, init_policy.sync_module_states
    )
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            module,
            mesh_info,
            post_forward_mesh_info,
            state._device,
            mp_policy,
            offload_policy,
        )

    _init_repr(module, mesh_info, post_forward_mesh_info)
    for module in managed_modules:
        module._is_fsdp_managed_module = True  # type: ignore[assignment]
        module._fsdp_use_orig_params = True  # type: ignore[assignment]
    # Place FSDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"FSDP{cls.__name__}", (FSDP, cls), dct)
    module.__class__ = new_cls
    return module


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> None:
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

    def extra_repr(self, *args: Any, **kwargs: Any):
        state = fully_shard.state(self)
        extra_repr_str = state._extra_repr_str if state is not None else ""
        return extra_repr_str + super().extra_repr(*args, **kwargs)  # type: ignore[misc]

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
            :meth:`wait` on the returned handle if the unshard op can be waited
            on in the module's pre-forward. FSDP will wait on the pending
            unshard op in the pre-forward automatically.
        """
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None or (
            (fsdp_param_group := state._fsdp_param_group) is None
        ):
            return None  # no-op
        fsdp_param_group._unshard(async_op=async_op)
        handle = UnshardHandle(cast(nn.Module, self))
        if async_op:
            return handle
        handle.wait()
        return None

    def reshard(self) -> None:
        """
        Reshards the module's parameters by freeing unsharded parameters if
        needed. This method is *not* recursive.
        """
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None or (
            (fsdp_param_group := state._fsdp_param_group) is None
        ):
            return  # no-op
        fsdp_param_group._reshard()

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
        module = cast(nn.Module, self)
        states = (
            [_get_module_fsdp_state(module)]
            if not recurse
            else [_get_module_fsdp_state(submodule) for submodule in module.modules()]
        )
        for state in states:
            if state and (fsdp_param_group := state._fsdp_param_group):
                fsdp_param_group.reduce_scatter_grads = requires_gradient_sync
                fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(self, requires_all_reduce: bool, recurse: bool = True):
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        module = cast(nn.Module, self)
        states = (
            [_get_module_fsdp_state(module)]
            if not recurse
            else [_get_module_fsdp_state(submodule) for submodule in module.modules()]
        )
        for state in states:
            if state and (fsdp_param_group := state._fsdp_param_group):
                fsdp_param_group.all_reduce_grads = requires_all_reduce

    # TODO: Expose a method for manually waiting on the gradient sync.
    def set_wait_for_gradient_sync(self, wait_for_gradient_sync: bool):
        """
        Sets if the root module should wait for the gradient synchronization
        (i.e. reduce-scatter and/or all-reduce) to finish at the end of
        backward. This can be disabled for some pipeline-parallel schedules to
        have more opportunity for communication/computation overlap. This is a
        no-op if called on a non-root module.
        """
        module = cast(nn.Module, self)
        state = _get_module_fsdp_state(module)
        if state is None or not state._is_root:
            raise AssertionError(f"Expects to be called on a root FSDP, not {module}")
        state._wait_for_grad_sync = wait_for_gradient_sync

    # We can properly allow `_apply` once we have: https://github.com/pytorch/pytorch/issues/113045
    def _apply(self, *args: Any, **kwargs):
        param_data_ptrs: List[int] = []
        param_devices: List[torch.device] = []
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is not None and (
            fsdp_param_group := state._fsdp_param_group
        ) is not None:
            fsdp_param_group._reshard()
            for fsdp_param in fsdp_param_group.fsdp_params:
                assert isinstance(fsdp_param.sharded_param, DTensor)  # mypy
                param_data_ptrs.append(
                    fsdp_param.sharded_param._local_tensor.data_ptr()
                )
                param_devices.append(fsdp_param.sharded_param.device)
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        new_param_data_ptrs: List[int] = []
        new_param_devices: List[torch.device] = []
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is not None and (
            fsdp_param_group := state._fsdp_param_group
        ) is not None:
            for fsdp_param in fsdp_param_group.fsdp_params:
                module_info = fsdp_param._module_info
                param = getattr(module_info.module, module_info.param_name)
                if not isinstance(param, DTensor):
                    raise RuntimeError(
                        f"An FSDP parameter has been changed to a non-DTensor on {self}"
                    )
                new_param_data_ptrs.append(param._local_tensor.data_ptr())
                new_param_devices.append(param.device)
        if len(param_data_ptrs) != len(new_param_data_ptrs):
            raise RuntimeError("Registered FSDP parameters changed")
        for ptr, device, new_ptr, new_device in zip(
            param_data_ptrs, param_devices, new_param_data_ptrs, new_param_devices
        ):
            if ptr != new_ptr or device != new_device:
                raise NotImplementedError(
                    "FSDP does not support _apply methods that change tensor storage or device. "
                    "Please call them before applying FSDP if possible or use the device argument."
                )
        return ret


class UnshardHandle:
    def __init__(self, module: nn.Module):
        self.module = module

    def wait(self):
        if (state := _get_module_fsdp_state(self.module)) is None or (
            (fsdp_param_group := state._fsdp_param_group) is None
        ):
            return  # no-op
        fsdp_param_group._wait_for_unshard()


def _init_repr(
    module: nn.Module,
    mesh_info: FSDPMeshInfo,
    post_forward_mesh_info: Optional[FSDPMeshInfo],
):
    """
    Initializes the string to use for ``extra_repr`` to add a sharding strategy
    tag plus some sharding/replication info to make the annotation visible
    when printing the module.
    """
    state = fully_shard.state(module)
    state._extra_repr = module.extra_repr

    # {FSDP, HSDP} x {reshard after forward}
    if isinstance(mesh_info, HSDPMeshInfo):
        strategy_str = f"(Shard={mesh_info.shard_mesh_size}, Replicate={mesh_info.replicate_mesh_size})"
    elif isinstance(mesh_info, FSDPMeshInfo):
        strategy_str = f"(Shard={mesh_info.shard_mesh_size})"
    else:
        strategy_str = ""
    if post_forward_mesh_info is None:
        reshard_str = "; No Reshard After Forward"
    else:
        if post_forward_mesh_info.shard_mesh_size != mesh_info.shard_mesh_size:
            reshard_str = (
                f"; After Forward: (Shard={post_forward_mesh_info.shard_mesh_size})"
            )
        else:
            reshard_str = ""

    def dp_extra_repr() -> str:
        return f"{strategy_str}{reshard_str}" + state._extra_repr()

    state._extra_repr_str = f"{strategy_str}{reshard_str}"
