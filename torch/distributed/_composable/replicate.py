import weakref
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract


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
        self._names: List[str] = []

    def mark_module(self, module: nn.Module, **kwargs) -> None:
        if _is_fully_sharded(module):
            raise AssertionError(
                "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
            )
        self.module = module
        replicate.state(module)._params_collected = False
        module.register_forward_pre_hook(self.forward_pre_hook)
        # TODO(@yhcharles): fix type error
        module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]
        self.kwargs = kwargs

    def _collect_params(self, module: nn.Module) -> None:
        # skip if managed by fully_sharded API
        if _is_fully_sharded(module):
            return

        if module in self.ignored_modules:
            return  # if module A is ignored, all of A's children are also ignored.

        self._param_list.extend(
            p for p in module.parameters(recurse=False) if p not in self.ignored_params
        )

        for child_module in module.children():
            self._collect_params(child_module)

    def init_helper(self) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True

        self._collect_params(self.module)
        # Only saved for testing
        replicate.state(self.module)._names = self._names
        if "device_ids" in self.kwargs:
            # replicate() supports a small usability enhancement where
            # device_ids=[self.device] can also be passed in for CPU so users
            # don't have to code change for CPU / GPU runs.
            device_ids = self.kwargs["device_ids"]
            if device_ids is not None and not isinstance(device_ids, list):
                raise RuntimeError(
                    f"Expected device_ids arg to be Optional[List[Union[int, torch.device]]], got {device_ids}"
                )
            device_id = device_ids[0]
            if isinstance(device_id, torch.device) and device_id.type == "cpu":
                self.kwargs["device_ids"] = None

        self._ddp = DistributedDataParallel(self._param_list, **self.kwargs)
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def forward_pre_hook(
        self, module: nn.Module, input: Tuple[torch.Tensor, ...]
    ) -> None:
        self.init_helper()
        self._ddp._pre_forward()

    def forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp._post_forward(output)
