# mypy: allow-untyped-defs
from contextlib import contextmanager, nullcontext
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator,
    _DEFAULT_DETERMINISM_MODE,
)

from .contract import contract


@contextmanager
def _no_hook(module: nn.Module):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    orig_enable_hook = checkpoint.state(module).enable_hook
    checkpoint.state(module).enable_hook = False
    try:
        yield
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook


@contract()
def checkpoint(module: nn.Module) -> nn.Module:
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if checkpoint.state(module).enable_hook:

            def context_fns():
                return nullcontext(), _no_hook(module)

            checkpoint.state(
                module
            )._ac_generator = _checkpoint_without_reentrant_generator(
                module, True, context_fns, _DEFAULT_DETERMINISM_MODE, False, *inputs
            )
            next(checkpoint.state(module)._ac_generator)

    def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            try:
                next(checkpoint.state(module)._ac_generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        #  Ensure that we no longer hold on to the generator. always_call=True helps ensure we
        # clear this even in the case of exception in fwd pass.
        checkpoint.state(module)._ac_generator = None

    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module
