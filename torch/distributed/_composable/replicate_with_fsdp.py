# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import overload

import torch
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _apply_fsdp_to_module,
    _get_device_from_mesh,
    _get_modules_and_states,
    _init_default_mesh,
    _init_param_group,
    _validate_module,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.fsdp._fully_shard._fully_shard import (
    _unimplemented_deepcopy,
    FSDPModule,
)
from torch.distributed.tensor import DeviceMesh

from .contract import _get_registry, contract


cls_to_replicate_cls: dict[type, type] = {}

_ROOT_MODULE_PREFIX = ""

logger = logging.getLogger("torch.distributed._composable.replicate_with_fsdp")


class _ReplicateStateContext:
    """This has state shared across Replicate states."""

    def __init__(self) -> None:
        # All Replicate states in the root state's module tree
        self.all_states: list[_ReplicateState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: _ReplicateState | None = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


def _get_module_replicate_state(module: nn.Module) -> _ReplicateState | None:
    """Checks if module state is ReplicateState"""
    state = _get_module_state(module)
    if isinstance(state, _ReplicateState):
        return state
    return None


class _ReplicateState(FSDPState):
    """
    Replicate state inherits from FSDP state and overrides the state name
    and module state getter.
    """

    _state_name: str = "Replicate"

    def __init__(self) -> None:
        super().__init__()
        self._state_ctx = _ReplicateStateContext()  # type: ignore[assignment]

    def _get_state_for_module(self, module: nn.Module) -> FSDPState | None:
        """Override to use replicate-specific state getter."""
        return _get_module_replicate_state(module)

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool = False,
    ) -> None:
        super().init(modules, device, mp_policy, auto_reshard_after_forward)


@overload
# pyrefly: ignore [inconsistent-overload]
def replicate(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
) -> ReplicateModule: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def replicate(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
) -> list[ReplicateModule]: ...


@contract(state_cls=_ReplicateState)  # type: ignore[misc]
def replicate(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: set[nn.Parameter] | None = None,
):
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed._composable.replicate_with_fsdp")

    _validate_module_for_replicate(module)

    mesh = mesh or _init_default_mesh(mesh_dim_names=("replicate",))
    if mesh.ndim != 1:
        raise ValueError(f"replicate expects a 1D DeviceMesh but got {mesh}")

    mesh_info = DDPMeshInfo(mesh, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)

    post_forward_mesh_info = None

    arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
        module,
        device,
        ignored_params,
        is_composable_fn=is_composable_with_replicate,
        get_state_fn=_get_module_replicate_state,
    )
    state = replicate.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy)

    _init_param_group(
        state,
        params,
        modules,
        mesh_info,
        post_forward_mesh_info,
        device,
        None,  # shard_placement_fn
        mp_policy,
        offload_policy,
    )

    # Place Replicate leftmost for highest priority in the method resolution order
    _apply_fsdp_to_module(
        modules, cls_to_replicate_cls, ReplicateModule, "Replicate", _unimplemented_deepcopy
    )
    return arg_module


class ReplicateModule(FSDPModule):
    # Index in MRO where the original class is found.
    # For Replicate: [Replicate<Orig>, ReplicateModule, FSDPModule, Orig, ...] -> index 3
    _orig_cls_mro_index: int = 3


def is_composable_with_replicate(module: nn.Module) -> bool:
    """Checks if replicate can be applied with module"""
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "fully_shard" not in registry


def _validate_module_for_replicate(module: nn.Module) -> None:
    """
    Validate that the module can be used with replicate.

    Raises RuntimeError if already managed by fully_shard.
    Raises ValueError if the module is a container that doesn't implement forward.
    """
    if not is_composable_with_replicate(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )
    _validate_module(module, "replicate")
