# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _insert_module_state
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    detect_compiled_autograd,
    HSDPMeshInfo,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    _register_group_forward_hooks,
    FSDPState,
)
from torch.distributed.fsdp._fully_shard._fully_shard import (
    _unimplemented_deepcopy,
    FSDPModule,
)
from torch.distributed.tensor import DeviceMesh, init_device_mesh
from torch.distributed.utils import _get_root_modules

from .contract import _get_registry, contract


if TYPE_CHECKING:
    from torch.distributed.tensor import Shard


cls_to_fsdp_cls: dict[type, type] = {}

_ROOT_MODULE_PREFIX = ""

logger = logging.getLogger("torch.distributed._composable.replicate_with_fsdp")


class _ReplicateStateContext:
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        # All FSDP states in the root state's module tree
        self.all_states: list[_ReplicateState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: Optional[_ReplicateState] = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: Optional[torch.Event] = None


def _get_module_replicate_state(module: nn.Module) -> Optional[_ReplicateState]:
    """Checks if module state is ReplicateState"""
    state = _get_module_state(module)
    if isinstance(state, _ReplicateState):
        return state
    return None


class _ReplicateState(FSDPState):
    """
    Replicate state functionality is adapted from FSDP state.
    In the future, could experiment with inheriting from it instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._state_ctx = _ReplicateStateContext()  # type: ignore[assignment]

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        self._auto_reshard_after_forward = auto_reshard_after_forward
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. Replicate has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(
                f"Replicate requires a single root module but got {self._modules}"
            )
        detect_compiled_autograd()
        root_module = self._modules[0]
        visited_states: set[_ReplicateState] = set()
        for module_name, module in root_module.named_modules():
            if (state := _get_module_replicate_state(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        "Replicate state has already been lazily initialized for "
                        f"{module_name}\nReplicate requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            self._state_ctx.all_states.append(state)
            visited_states.add(state)
        if self._fsdp_param_group and self._auto_reshard_after_forward:
            # For the root, do not reshard after forward since for training,
            # the parameters would be freed and all-gathered immediately
            self._fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:  # type: ignore[assignment]
            if state._fsdp_param_group:  # type: ignore[union-attr]
                state._fsdp_param_group.lazy_init()  # type: ignore[union-attr]


def replicate_impl(
    module,
    mesh: DeviceMesh,
    *,
    device_id: Optional[Union[int, torch.device]] = None,
    reshard_after_forward: Optional[Union[bool, int]] = None,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
):
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"replicate does not support containers that do not implement forward: {module}"
        )

    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim != 2:
        raise ValueError(f"replicate expects a 2D DeviceMesh but got {mesh}")

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
    state = replicate.state(modules[0])  # type: ignore[attr-defined] # see [1]
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

    # Place Replicate leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    **kwargs,
) -> nn.Module:
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """

    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                "Expected device_id to be int or torch.device, "
                f"but got {type(kwargs['device_id'])}"
            )

    if not is_composable_with_replicate(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )

    device_mesh = kwargs.pop("device_mesh", None)
    if device_mesh is not None:
        from torch.distributed.device_mesh import _mesh_resources

        root_mesh = _mesh_resources.get_root_mesh(device_mesh)
        # if a root mesh is not the same as device_mesh,
        # meaning the device_mesh is sliced out from the root mesh.
        if root_mesh != device_mesh:
            # TODO: This is a temporary work around to enable DDP + TP.
            # We should do the logic in DDP so that the 2D implementation is
            # sound and the state_dict works out of the box.
            #
            # This won't conflict with what is done in DDP class as the module
            # replicate is going to pass is NOT the original module.
            from torch.distributed.tensor.parallel.ddp import (
                _localize_dtensor,
                _reconstruct_dtensor,
            )

            module.register_forward_pre_hook(_reconstruct_dtensor)
            module.register_forward_hook(_localize_dtensor)
    else:
        device_mesh = replicate_mesh()

    module = replicate_impl(module, mesh=device_mesh, **kwargs)
    return module


def _get_managed_modules(
    root_modules: tuple[nn.Module, ...],
    ignored_params: Optional[set[nn.Parameter]] = None,
) -> list[nn.Module]:
    modules: list[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``replicate`` already applied.
        """
        if not is_composable_with_replicate(module):
            return
        elif (
            module not in root_modules_set
            and _get_module_replicate_state(module) is not None
        ):
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


def is_composable_with_replicate(module: nn.Module) -> bool:
    """Checks if replicate can be applied with module"""
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "fully_shard" not in registry


def replicate_mesh():
    """Creates a device mesh for replicate if the user doesn't provide one"""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device = torch._C._get_accelerator()
    mesh = init_device_mesh(
        device.type,
        mesh_shape=(default_pg.size(), 1),
        mesh_dim_names=("replicate", "shard"),
    )
    return mesh


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


def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    """
    Decide if it is safe to ignore a module for applying replicate.
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
