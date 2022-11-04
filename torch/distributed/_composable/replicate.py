from typing import List, Tuple

import torch
import torch.nn as nn

from . import _ddp
from .contract import contract


class DistributedState:
    ...


class ReplicateState(DistributedState):
    def __init__(self) -> None:
        self.modules: List[nn.Module] = []
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()

    def mark_modules(self, *modules: nn.Module) -> None:
        for module in modules:
            self.modules.append(module)
            replicate.state(module)._distributed_state = self
            replicate.state(module)._params_collected = False

    def _recursive_collect_params(self, module: nn.Module) -> None:
        # TODO: skip if managed by other APIs

        if replicate.state(module) is None or getattr(
            replicate.state(module), "_param_collected", False
        ):
            return
        replicate.state(module)._param_collected = True

        self._param_list.extend(
            param for param in module.parameters() if param.requires_grad
        )
        for child in module.children():
            self._recursive_collect_params(child)

    def init_helper(self):
        if self.has_initialized:
            return

        self.has_initialized = True
        for module in self.modules:
            self._recursive_collect_params(module)

        self._ddp = _ddp.DistributedDataParallel(self._param_list)

    def root_module_forward_pre_hook(
        self, module: nn.Module, input: Tuple[torch.Tensor]
    ) -> None:
        self.init_helper()
        self._ddp.pre_forward()

    def root_module_forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp.post_forward(output)


# TODO(@yhcharles): use a per-model instance instead of a global one
_default_state = ReplicateState()


@contract
def replicate(
    module: nn.Module,  # NOTE: contract now supports single module only
    dist_state: ReplicateState = _default_state,
) -> nn.Module:
    r"""Replicates module(s)

    Args:
        modules (torch.nn.Module): modules to replicate

    Example::
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    dist_state.mark_modules(module)
    return module


def mark_root_module(
    module: nn.Module, dist_state: ReplicateState = _default_state
) -> nn.Module:
    r"""Mark the root module. Its sub-modules can be replicated.

    Args:
        modules (torch.nn.Module): root module

    Example::
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    module.register_forward_pre_hook(dist_state.root_module_forward_pre_hook)
    # TODO(@yhcharles): fix type error
    module.register_forward_hook(
        dist_state.root_module_forward_post_hook  # type: ignore[arg-type]
    )
    return module
