import itertools
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh, DTensor, init_device_mesh
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from ._fsdp_common import (
    _is_composable_with_fsdp,
    DataParallelMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
)
from ._fsdp_state import _get_module_fsdp_state


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
    from ._fsdp_state import FSDPState


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


def _validate_module(module: nn.Module, func_name: str) -> None:
    """
    Validate that the module can be used with fully_shard or replicate.

    Raises ValueError if the module is a container that doesn't implement forward.
    """
    if (
        isinstance(module, (nn.ModuleList, nn.ModuleDict))
        and module.__class__.forward is nn.Module.forward
    ):
        raise ValueError(
            f"{func_name} does not support containers that do not implement forward: {module}"
        )


def _validate_mesh(mesh: "DeviceMesh") -> None:
    """
    Validate that the mesh can be used with fully_shard.

    Raises ValueError if the mesh is not 1D or 2D.
    Raises AssertionError if the mesh is 2D but mesh_dim_names is not specified.
    """
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    if mesh.ndim == 2 and mesh.mesh_dim_names is None:
        raise AssertionError(
            "Please init the 2D mesh for HSDP with mesh_dim_names specified"
        )


def _get_mesh_info(mesh: "DeviceMesh") -> "FSDPMeshInfo":
    """
    Get the appropriate mesh info for the given mesh.

    Returns FSDPMeshInfo for 1D mesh, HSDPMeshInfo for 2D mesh.
    """
    if mesh.ndim == 1:
        return FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        return HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)


def _get_post_forward_mesh_info(
    reshard_after_forward: bool | int, mesh_info: FSDPMeshInfo
) -> FSDPMeshInfo | None:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            msg = (
                "reshard_after_forward=1 (int) means resharding parameters to world size 1, "
                "instead of reshard_after_forward=True (bool)"
            )
            warning_once(logger, msg, stacklevel=2)
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(
            mesh_info.mesh.device_type, post_forward_mesh_tensor
        )
        post_forward_mesh_info = HSDPMeshInfo(
            post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0
        )
    return post_forward_mesh_info


def _init_default_mesh(
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device = torch._C._get_accelerator()
    mesh = init_device_mesh(
        device.type,
        mesh_shape=(default_pg.size(),),
        mesh_dim_names=mesh_dim_names,
    )
    return mesh


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    return _init_default_mesh()


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    """
    Decide if it is safe to ignore a module for applying fully_shard.
    """
    if module in ignore_decision:
        return ignore_decision[module]

    if len(list(module.buffers(recurse=False))) > 0:
        # Cannot ignore a module with any buffer
        ignore_decision[module] = False
        return False

    for _, param in module.named_parameters(recurse=False):
        if param not in ignored_params:
            # at least one param is not ignored. So this module shouldn't be.
            ignore_decision[module] = False
            return False

    # Need to consider descendants of module
    for child in list(module.children()):
        ignore_child = _ignore_module(child, ignored_params, ignore_decision)
        if not ignore_child:
            # Cannot ignore module if one of its children is not ignored
            ignore_decision[module] = False
            return False

    # Safe to ignore module
    ignore_decision[module] = True
    return True


def _adjust_managed_modules(
    modules: list[nn.Module], ignored_params: set[nn.Parameter]
) -> list[nn.Module]:
    """
    Adjust the given list of managed modules by removing those with all parameters ignored.
    """
    ignore_decision: dict[nn.Module, bool] = {}
    new_modules = []
    for module in modules:
        ignored = _ignore_module(module, ignored_params, ignore_decision)
        if not ignored:
            new_modules.append(module)
    return new_modules


def _get_managed_modules(
    root_modules: tuple[nn.Module, ...],
    ignored_params: set[nn.Parameter] | None = None,
    is_composable_fn: "Callable[[nn.Module], bool] | None" = None,
    get_state_fn: "Callable[[nn.Module], Any] | None" = None,
) -> list[nn.Module]:
    """
    Get the list of managed modules for FSDP/replicate.

    Args:
        root_modules: The root modules to start the search from.
        ignored_params: Parameters to ignore.
        is_composable_fn: Callable to check if a module is composable.
            Defaults to ``_is_composable_with_fsdp``.
        get_state_fn: Callable to get the state of a module.
            Defaults to ``_get_module_fsdp_state``.
    """
    if is_composable_fn is None:
        is_composable_fn = _is_composable_with_fsdp
    if get_state_fn is None:
        get_state_fn = _get_module_fsdp_state

    modules: list[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        if not is_composable_fn(module):
            return
        elif module not in root_modules_set and get_state_fn(module) is not None:
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    for root_module in root_modules:
        dfs(root_module)

    if ignored_params is None:
        return modules

    adjusted_modules = _adjust_managed_modules(modules, ignored_params)
    return adjusted_modules


def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. "
            f"Change {name} to a 1D tensor with numel equal to 1."
        )


def _get_managed_states(
    modules: list[nn.Module], ignored_params: set[nn.Parameter] | None = None
) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    params: list[nn.Parameter] = []
    buffers: list[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: set[nn.Parameter] = set()
    visited_buffers: set[torch.Tensor] = set()
    if ignored_params is None:
        ignored_params = set()

    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param in ignored_params:
                # do not include an ignored parameters
                continue
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers


def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
    # Follow the logic in `nn.Module._apply`
    # pyrefly: ignore [bad-argument-type]
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device or tensor.device.type == "meta":
            # Keep meta-device tensors on meta device for deferred init
            continue
        if isinstance(tensor, DTensor):
            if (dtensor_mesh_type := tensor.device_mesh.device_type) != device.type:
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        tensor_ = tensor
        if is_traceable_wrapper_subclass(tensor_):
            with torch.no_grad():  # avoid autograd increasing C++ refcount by 1
                tensor_on_device = nn.Parameter(tensor.to(device))
            torch.utils.swap_tensors(tensor, tensor_on_device)
        else:
            tensor.data = tensor.to(device)


def _apply_to_module(
    modules: tuple[nn.Module, ...],
    cls_to_wrapper_cls: dict[type, type],
    wrapper_module_cls: type,
    wrapper_cls_prefix: str,
    unimplemented_deepcopy: "Callable",
) -> None:
    """
    Modify module classes to include the wrapper class in their MRO.

    Args:
        modules: The modules to apply the wrapper to.
        cls_to_wrapper_cls: Cache dict mapping original class to wrapper class.
        wrapper_module_cls: The wrapper module class (e.g., FSDPModule, ReplicateModule).
        wrapper_cls_prefix: Prefix for the dynamically created class name (e.g., "FSDP", "Replicate").
        unimplemented_deepcopy: The deepcopy function to use for the wrapper class.
    """
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_wrapper_cls.get(cls)
        if not new_cls:
            dct = {"__deepcopy__": unimplemented_deepcopy}
            new_cls = type(
                f"{wrapper_cls_prefix}{cls.__name__}", (wrapper_module_cls, cls), dct
            )
            cls_to_wrapper_cls[cls] = new_cls
        module.__class__ = new_cls


def _init_param_group(
    state: "FSDPState",
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: FSDPMeshInfo | None,
    device: torch.device,
    shard_placement_fn: "Callable[[nn.Parameter], Any] | None",
    mp_policy: "MixedPrecisionPolicy",
    offload_policy: "OffloadPolicy",
) -> None:
    """
    Initialize the FSDP param group for the given state if there are params.

    This is shared between fully_shard and replicate.
    """
    # Import here to avoid circular imports
    from ._fsdp_param_group import FSDPParamGroup

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


def _get_modules_and_states(
    module: nn.Module,
    device: torch.device,
    ignored_params: set[nn.Parameter] | None,
    is_composable_fn: "Callable[[nn.Module], bool] | None" = None,
    get_state_fn: "Callable[[nn.Module], Any] | None" = None,
) -> tuple[
    nn.Module,
    tuple[nn.Module, ...],
    list[nn.Module],
    list[nn.Parameter],
    list[torch.Tensor],
]:
    """
    Get modules tuple, managed modules, params, and buffers for FSDP/replicate initialization.

    Returns:
        Tuple of (arg_module, modules, managed_modules, params, buffers)
    """
    from torch.distributed.utils import _get_root_modules

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )

    managed_modules = _get_managed_modules(
        modules, ignored_params, is_composable_fn, get_state_fn
    )
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)

    return arg_module, modules, managed_modules, params, buffers
