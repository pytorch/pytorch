import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract

_ROOT_MODULE_PREFIX = ""


@contract()
def replicate(
    module: nn.Module,  # NOTE: contract now supports single module only
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
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
    torch._C._log_api_usage_once("torch.distributed.replicate")
    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                f"Expected device_id to be int or torch.device, but got {type(kwargs['device_id'])}"
            )
    _ReplicateState(ignored_modules=ignored_modules).mark_module(module, **kwargs)
    return module


def _is_fully_sharded(module: nn.Module) -> bool:
    r"""Check if module is marked with fully_shard."""
    return "fully_shard" in _get_registry(module)


class _ReplicateState:
    def __init__(self, ignored_modules: Optional[Iterable[torch.nn.Module]]) -> None:
        self.module: Optional[nn.Module] = None
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        self.kwargs: dict = {}
        self.ignored_modules: Set[torch.nn.Module] = (
            set(ignored_modules) if ignored_modules is not None else set()
        )
        self.ignored_params: Set[torch.nn.Parameter] = {
            p for m in self.ignored_modules for p in m.parameters()
        }
        # Only used for testing
        self._param_names: List[str] = []

    def mark_module(self, module: nn.Module, **kwargs) -> None:
        if _is_fully_sharded(module):
            raise AssertionError(
                "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
            )
        self.module = module
        replicate.state(module)._params_collected = False
        module.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        # TODO(@yhcharles): fix type error
        module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]
        self.kwargs = kwargs

    def _collect_params(
        self, module: nn.Module, prefix: str = _ROOT_MODULE_PREFIX
    ) -> None:
        # skip if managed by fully_sharded API
        if _is_fully_sharded(module):
            return

        if module in self.ignored_modules:
            return  # if module A is ignored, all of A's children are also ignored.

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in self.ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(module=child_module, prefix=f"{recurse_prefix}{name}")

    def init_helper(self) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True

        self._collect_params(self.module)  # type: ignore[arg-type]
        # Only saved for testing
        replicate.state(self.module)._replicate_param_names = self._param_names
        if "device_id" in self.kwargs:
            # replicate() supports a small usability enhancement where
            # user can pass in device_id as a Union[int, torch.device] even for
            # CPU devices so users don't have to change code for CPU/GPU runs.
            # We derive the right device_ids to feed into DDP to support this.
            if self.kwargs["device_id"] is not None:
                device_id = self.kwargs["device_id"]
                # Convert to device_ids that DDP expects.
                if isinstance(device_id, torch.device) and device_id.type == "cpu":
                    # CPU modules receive device_ids None
                    self.kwargs["device_ids"] = None
                else:
                    # GPU modules expect device_ids=[cuda_device]
                    self.kwargs["device_ids"] = [device_id]
            else:
                self.kwargs["device_ids"] = None
            self.kwargs.pop("device_id")

        self._ddp = DistributedDataParallel(self._param_list, **self.kwargs)
        # Weakref to the DDP instance is currently only used for testing.
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def forward_pre_hook(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        self.init_helper()
        args, kwargs = self._ddp._pre_forward(*args, **kwargs)
        return args, kwargs

    def forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp._post_forward(output)
