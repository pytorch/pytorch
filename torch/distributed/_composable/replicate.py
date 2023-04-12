from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel

from .contract import contract


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


class _ReplicateState:
    def __init__(self, ignored_modules) -> None:
        self.module = None
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        self.kwargs: dict = {}
        self.ignored_modules = (
            set(ignored_modules) if ignored_modules is not None else set()
        )

    def mark_module(self, module: nn.Module, **kwargs) -> None:
        self.module = module
        replicate.state(module)._params_collected = False
        module.register_forward_pre_hook(self.forward_pre_hook)
        # TODO(@yhcharles): fix type error
        module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]
        self.kwargs = kwargs

    def _collect_params(self, module: nn.Module) -> None:

        if module in self.ignored_modules:
            return  # if module A is ignored, all of A's children are also ignored.

        ignored_params: Set[torch.nn.Parameter] = {
            p for m in self.ignored_modules for p in m.parameters()
        }
        data_parallel_param_tup = [(n, p) for n, p in module.named_parameters() if p not in ignored_params]
        names, params = zip(*data_parallel_param_tup)
        # _names field is just for testing.
        replicate.state(module)._names = names
        self._param_list.extend(params)

    def init_helper(self) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True

        self._collect_params(self.module)

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
