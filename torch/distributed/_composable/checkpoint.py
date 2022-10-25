import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import detach_variable

from contextlib import contextmanager
from typing import Any, Optional, Tuple

from .contract import contract


# state key to store per-module enable_hook flag
ENALBE_HOOK_KEY = object()


@contextmanager
def _no_hook(module: nn.Module):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    orig_enable_hook = checkpoint.get_state(module, ENALBE_HOOK_KEY)
    checkpoint.set_state(module, ENALBE_HOOK_KEY, False)
    try:
        yield
    except Exception:
        raise
    finally:
        checkpoint.set_state(module, ENALBE_HOOK_KEY, orig_enable_hook)


class _ModuleHookCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: nn.Module, output: Any, *inputs) -> Any:
        ctx.module = module

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                tensor_inputs.append(inp)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(inp)

        ctx.save_for_backward(*tensor_inputs)

        return output

    @staticmethod
    def backward(
        ctx, output_grads: Tuple[Optional[torch.Tensor]]
    ) -> Tuple[Optional[torch.Tensor]]:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an "
                "`inputs` parameter is passed to .backward(). Please use "
                ".backward() and do not pass its `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        detached_inputs = detach_variable(tuple(inputs))
        with torch.enable_grad(), _no_hook(ctx.module):
            outputs = ctx.module(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        if isinstance(output_grads, torch.Tensor):
            output_grads = (output_grads,)

        # run backward() with only tensor that requires grad
        outputs_requires_grad = []
        output_grad_tensors = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_requires_grad.append(outputs[i])
                output_grad_tensors.append(output_grads[i])
        if len(outputs_requires_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_requires_grad, output_grad_tensors)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        # The two None is for forward argument module and output respectively.
        return (None, None) + grads


@contract
def checkpoint(module: nn.Module) -> nn.Module:
    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any]) -> None:
        if checkpoint.get_state(module, ENALBE_HOOK_KEY):
            torch.set_grad_enabled(False)

    def forward_hook(module: nn.Module, inputs: Tuple[Any], output: Any) -> Any:
        if not checkpoint.get_state(module, ENALBE_HOOK_KEY):
            return output

        torch.set_grad_enabled(True)
        return _ModuleHookCheckpointFunction.apply(module, output, *inputs)

    # This hook does the following things:
    # 1. detach outputs from the autograd graph to discard activations
    # 2. insert an autograd.Function after the forward pass to recompute
    #    activations during the backward pass.
    checkpoint.set_state(module, ENALBE_HOOK_KEY, True)
    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_hook)
    return module
