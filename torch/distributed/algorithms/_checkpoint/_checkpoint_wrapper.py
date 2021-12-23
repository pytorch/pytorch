from enum import Enum, auto

import torch
from torch.utils.checkpoint import checkpoint


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class _CheckpointWrapper(torch.nn.Module):
    """
    An nn.Module that wraps another nn.Module with checkpointing.
    """
    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
    ):
        super().__init__()
        self.mod = mod
        self.checkpoint_impl = checkpoint_impl

    def forward(self, *args, **kwargs):
        return checkpoint(
            self.mod,
            use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
            *args,
            **kwargs,
        )


def checkpoint_wrapper(
    module: torch.nn.Module, checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT
) -> torch.nn.Module:
    """
    A convenience wrapper for activation checkpointing. If the module is wrapped
    with this function, all subsequent calls to the module will automatically
    perform checkpointing without the user having to explicitly call ``checkpoint``
    function.
    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Currently only
            CheckpointImpl.REENTRANT is supported.
    Returns:
        (nn.Module):
            Wrapped module
    """
    # saved tensor hooks based-checkpoint wrapper is not yet supported.
    if checkpoint_impl == CheckpointImpl.NO_REENTRANT:
        raise ValueError(
            "No support for non-reentrant based checkpoint implementation."
        )

    return _CheckpointWrapper(module, checkpoint_impl)
