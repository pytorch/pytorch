# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import overload, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _apply_to_module,
    _get_device_from_mesh,
    _get_modules_and_states,
    _init_default_mesh,
    _init_param_group,
    _validate_module as _validate_module_common,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState, FSDPStateContext
from torch.distributed.fsdp._fully_shard._fully_shard import (
    _unimplemented_deepcopy,
    FSDPModule,
)

from .contract import _get_registry, contract


if TYPE_CHECKING:
    from torch.distributed.tensor import DeviceMesh


cls_to_replicate_cls: dict[type, type] = {}

logger = logging.getLogger("torch.distributed._composable.replicate_with_fsdp")


class _ReplicateStateContext(FSDPStateContext["_ReplicateState"]):
    """
    State shared across Replicate states.

    This is a typed subclass of FSDPStateContext parameterized with _ReplicateState,
    providing correct type annotations (e.g., all_states: list[_ReplicateState]).
    It also allows call sites to differentiate between Replicate and FSDP contexts
    via isinstance checks if needed.
    """


def _get_module_replicate_state(module: nn.Module) -> _ReplicateState | None:
    state = _get_module_state(module)
    if isinstance(state, _ReplicateState):
        return state
    return None


class _ReplicateState(FSDPState):
    _state_name: str = "Replicate"

    def __init__(self) -> None:
        super().__init__()
        self._state_ctx = _ReplicateStateContext()

    def _get_state_for_module(self, module: nn.Module) -> FSDPState | None:
        return _get_module_replicate_state(module)

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
    _validate_module(module)
    mesh = mesh or _init_default_mesh(mesh_dim_names=("replicate",))
    _validate_mesh(mesh)
    mesh_info = DDPMeshInfo(mesh, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    # managed_modules (3rd return) and buffers (5th return) are unused:
    # - managed_modules: FSDP uses this to set Dynamo-specific attributes
    #   (_is_fsdp_managed_module, _fsdp_use_orig_params), which replicate doesn't need
    # - buffers: already moved to device by _get_modules_and_states; replicate
    #   doesn't need to track them separately
    arg_module, modules, _, params, _ = _get_modules_and_states(
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
        None,  # post_forward_mesh_info
        device,
        None,  # shard_placement_fn
        mp_policy,
        offload_policy,
    )

    # Place Replicate leftmost for highest priority in the method resolution order
    _apply_to_module(
        modules,
        cls_to_replicate_cls,
        ReplicateModule,
        "Replicate",
        _unimplemented_deepcopy,
    )
    return arg_module


class ReplicateModule(FSDPModule):
    # Index in MRO where the original class is found.
    # For Replicate: [Replicate<Orig>, ReplicateModule, FSDPModule, Orig, ...] -> index 3
    _orig_cls_mro_index: int = 3


def is_composable_with_replicate(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    return "fully_shard" not in registry


def _validate_module(module: nn.Module) -> None:
    if not is_composable_with_replicate(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )
    _validate_module_common(module, "replicate")


def _validate_mesh(mesh: DeviceMesh) -> None:
    if mesh.ndim != 1:
        raise ValueError(f"replicate expects a 1D DeviceMesh but got {mesh}")
