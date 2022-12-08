from typing import List, Tuple

import torch
import torch.nn as nn

from . import _ddp
from .contract import contract


class _ReplicateState:
    def __init__(self) -> None:
        self.modules: List[nn.Module] = []
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        self.kwargs: dict = {}

    def mark_modules(self, *modules: nn.Module, **kwargs) -> None:
        for module in modules:
            self.modules.append(module)
            replicate.state(module)._distributed_state = self
            replicate.state(module)._params_collected = False
            module.register_forward_pre_hook(self.forward_pre_hook)
            # TODO(@yhcharles): fix type error
            module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]
        self.kwargs = kwargs

    def _recursive_collect_params(self, module: nn.Module) -> None:
        # TODO: skip if managed by other APIs

        if hasattr(replicate.state(module), "_params_collected"):
            if replicate.state(module)._params_collected:
                return
            replicate.state(module)._params_collected = True

        self._param_list.extend(
            param
            for param in module.parameters(recurse=False)
            # for param in module.parameters()
            if param.requires_grad
        )
        for child in module.children():
            self._recursive_collect_params(child)

    def init_helper(self) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True
        for module in self.modules:
            self._recursive_collect_params(module)

        self._ddp = _ddp.DistributedDataParallel(
            self._param_list, **self.kwargs
        )

    def forward_pre_hook(
        self, module: nn.Module, input: Tuple[torch.Tensor]
    ) -> None:
        self.init_helper()
        self._ddp.pre_forward()

    def forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp.post_forward(output)


@contract
def replicate(
    module: nn.Module,  # NOTE: contract now supports single module only
    **kwargs,
) -> nn.Module:
    r"""Replicates module(s)

    Args:
        modules (torch.nn.Module): modules to replicate

    Example::
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    _ReplicateState().mark_modules(module, **kwargs)
    return module
