# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import logging
from dataclasses import replace
from typing import Any, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs

from .._flex_shard.module_state import ModuleStateContext, ShardedModuleState
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _dynamo_disable


if TYPE_CHECKING:
    from .._flex_shard.module_state import BucketGroup
    from ._fsdp_engine import FullyShardBucket
    from ._fsdp_param import FSDPParam


_StateType = TypeVar("_StateType", bound="FSDPState")


class FSDPStateContext(ModuleStateContext[_StateType]):
    """The state shared across the FSDP states under one root."""


class FSDPState(ShardedModuleState):
    """The state of a ``fully_shard`` call on the engine's module lifecycle
    (``ShardedModuleState``), with FSDP2's frontend rules: the root moves its
    forward inputs to its device, and FSDP parameters get their FQNs."""

    _state_name: str = "FSDP"
    _logger: logging.Logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

    def __init__(self) -> None:
        super().__init__()
        self._state_ctx = FSDPStateContext()

    @property
    def _fsdp_param_groups(self) -> "list[FullyShardBucket]":
        """The engine buckets of this state's ``fully_shard`` call, one per
        mesh (per-param mesh support)."""
        return self._groups  # type: ignore[return-value]

    def _get_state_for_module(self, module: nn.Module) -> "FSDPState | None":
        """Get the state for a module. Subclasses can override to use different state getters."""
        return _get_module_fsdp_state(module)

    # Define a separate init since `__init__` is called in the contract
    def init(  # type: ignore[override]
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = replace(mp_policy, param_dtype_override_fn=None)
        input_dtype = (
            mp_policy.param_dtype
            if mp_policy.cast_forward_inputs and mp_policy.param_dtype
            else None
        )
        super().init(
            modules,
            device,
            (input_dtype, mp_policy.output_dtype),
            auto_reshard_after_forward,
        )

    def _root_inputs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self._device.type in [
            "cuda",
            "hpu",
            "xpu",
            "mtia",
            torch._C._get_privateuse1_backend_name(),
        ]:
            if args or kwargs:
                with torch.profiler.record_function("FSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args = args_tuple[0]
                kwargs = kwargs_tuple[0]
        return args, kwargs

    def _on_lazy_init(self, root_module: nn.Module) -> None:
        param_to_fsdp_param: dict[nn.Parameter, FSDPParam] = {
            fsdp_param.sharded_param: fsdp_param
            for state in self._state_ctx.all_states
            for group in state._groups
            for fsdp_param in group.fsdp_params  # type: ignore[attr-defined]
        }
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        seen: set[int] = set()
        for fsdp_param in param_to_fsdp_param.values():
            if fsdp_param._orig_param_uid in seen:
                raise ValueError(
                    f"Parameter '{fsdp_param._param_fqn}' is shared with a "
                    f"parameter already managed by another FSDP group. "
                    f"For shared/tied parameters, use "
                    f"fully_shard([module_a, module_b]) to place them in "
                    f"the same FSDP group."
                )
            seen.add(fsdp_param._orig_param_uid)

    # FSDP2's hooks run outside Dynamo
    @_dynamo_disable
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return super()._pre_forward(module, args, kwargs)

    @_dynamo_disable
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        return super()._post_forward(module, input, output)

    def _after_group_pre_forward(self, group: "BucketGroup") -> None:
        if group.has_spmd_types:  # type: ignore[attr-defined]
            for fsdp_param in group.fsdp_params:  # type: ignore[attr-defined]
                fsdp_param._restore_spmd_types(fsdp_param.unsharded_param)


def _get_module_fsdp_state(module: nn.Module) -> FSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None
