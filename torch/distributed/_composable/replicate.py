from typing import List, Tuple

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from .contract import _get_registry, contract


@contract()
def replicate(
    module: nn.Module,  # NOTE: contract now supports single module only
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
    _ReplicateState().mark_module(module, **kwargs)
    return module


def _can_compose(module: nn.Module) -> bool:
    r"""Check if module is composable for `replicate` API."""
    return "fully_shard" not in _get_registry(module)


class _ReplicateState:
    def __init__(self) -> None:
        self.module = None
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        self.kwargs: dict = {}

    def mark_module(self, module: nn.Module, **kwargs) -> None:
        if not _can_compose(module):
            raise AssertionError(
                "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
            )
        self.module = module
        replicate.state(module)._params_collected = False
        module.register_forward_pre_hook(self.forward_pre_hook)
        # TODO(@yhcharles): fix type error
        module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]
        self.kwargs = kwargs

    def _recursive_collect_params(self, module: nn.Module) -> None:
        # skip if managed by other APIs
        if not _can_compose(module):
            return

        # skip if module parameters already collected
        replicate_state = replicate.state(module)
        # if replicate_state is None, `module` is a child module that has not been explicitly
        # tagged as replicate().
        if replicate_state is not None:
            if hasattr(replicate_state, "_params_collected"):
                if replicate.state(module)._params_collected:
                    return
                replicate.state(module)._params_collected = True

        self._param_list.extend(
            param for param in module.parameters(recurse=False) if param.requires_grad
        )
        for child in module.children():
            self._recursive_collect_params(child)

    def init_helper(self) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True

        self._recursive_collect_params(self.module)

        self._ddp = DistributedDataParallel(self._param_list, **self.kwargs)

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
