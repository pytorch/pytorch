from enum import Enum, auto
import torch
from torch.utils.checkpoint import checkpoint
from typing import Any
from functools import partial
from weakref import ref


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT
):
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
    # Use weakref to avoid creating a refcycle: m -> m.forward -> m. This would
    # leak GPU memory because python won't gc the module when the module is
    # freed.
    module.forward = partial(  # type: ignore[assignment]
        _checkpointed_forward,
        type(module).forward,
        ref(module),
        checkpoint_impl,
    )
    return module

def _checkpointed_forward(
    original_forward: Any, weak_self: Any, checkpoint_impl: Any, *args: Any, **kwargs: Any
) -> Any:
    module = weak_self()
    # If grads are disabled, call into original forward
    if not torch.is_grad_enabled():
        return original_forward(module, *args, **kwargs)

    forward_args = (module, ) + args
    return checkpoint(
        original_forward,
        use_reentrant=(checkpoint_impl == CheckpointImpl.REENTRANT),
        *forward_args,
        **kwargs
    )
