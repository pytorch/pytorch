from enum import Enum, auto
from contextlib import suppress

import torch
from torch.autograd.graph import save_on_cpu
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
        offload_to_cpu: bool = False,
    ):
        super().__init__()
        self.mod = mod
        self.checkpoint_impl = checkpoint_impl
        self.offload_to_cpu = offload_to_cpu

    def forward(self, *args, **kwargs):
        offload_mgr = save_on_cpu(pin_memory=True) if self.offload_to_cpu else suppress()
        with offload_mgr:  # type: ignore[attr-defined]
            return checkpoint(
                self.mod,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                *args,
                **kwargs,
            )


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
    offload_to_cpu: bool = False,
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
        offload_to_cpu (Optional[bool]):
            Whether to offload outer activations to CPU. Note that this
            currently only works with CheckpointImpl.REENTRANT.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # saved tensor hooks based-checkpoint wrapper is not yet supported.
    if checkpoint_impl == CheckpointImpl.NO_REENTRANT:
        raise ValueError(
            "No support for non-reentrant based checkpoint implementation."
        )

    if offload_to_cpu and checkpoint_impl != CheckpointImpl.REENTRANT:
        raise ValueError(
            "No support for CPU offload activations and non-reentrant based "
            "checkpoint implementation."
        )

    return _CheckpointWrapper(module, checkpoint_impl, offload_to_cpu)
