import torch
import torch.nn as nn
from . import _ddp

from typing import List, Tuple


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
            # TODO(@yhcharles): do not directly set attribute
            module._distributed_state = self  # type: ignore[assignment]

    def _recursive_collect_params(self, module: nn.Module) -> None:
        if (
            getattr(module, "_distributed_state", None) is not None
        ):  # managed by another API
            return

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


def replicate(
    *modules: nn.Module, dist_state: ReplicateState = _default_state
) -> None:
    r"""Replicates module(s)

    Args:
        modules (torch.nn.Module): modules to replicate

    Example::
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    dist_state.mark_modules(*modules)


def mark_root_module(
    module: nn.Module, dist_state: ReplicateState = _default_state
) -> None:
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
