# mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator,
    _DEFAULT_DETERMINISM_MODE,
)

from .contract import _State, contract


@contextmanager
def _no_hook(module: nn.Module, user_ctx: AbstractContextManager | None = None):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """

    with user_ctx if user_ctx else nullcontext():
        orig_enable_hook = checkpoint.state(module).enable_hook
        checkpoint.state(module).enable_hook = False
        try:
            yield
        finally:
            checkpoint.state(module).enable_hook = orig_enable_hook


class _CheckpointState(_State):
    enable_hook: bool = False
    _ac_generator: Generator[None, None, None] | None


@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
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
        >>>     def __init__(self) -> None:
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

    use_reentrant = kwargs.pop("use_reentrant", False)
    if use_reentrant:
        raise NotImplementedError(
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead."
        )
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    user_context_fns = kwargs.pop("context_fn", None)
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)
    early_stop = kwargs.pop("early_stop", True)

    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def forward_pre_hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if checkpoint.state(module).enable_hook:

            def context_fns():
                if user_context_fns is not None:
                    ctx1, ctx2 = user_context_fns()
                    return ctx1, _no_hook(module, ctx2)
                else:
                    return nullcontext(), _no_hook(module)

            gen = _checkpoint_without_reentrant_generator(
                module,
                preserve_rng_state,
                context_fns,
                determinism_check,
                debug,
                early_stop,
                *args,
                **kwargs,
            )
            checkpoint.state(module)._ac_generator = gen
            next(gen)

    def forward_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            try:
                gen = checkpoint.state(module)._ac_generator
                assert gen is not None
                next(gen)
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
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module
