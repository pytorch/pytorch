import torch
import torch.nn as nn
from torch.utils.checkpoint import detach_variable

from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

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
    except Exception:
        raise
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook


class _ModuleHookCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: nn.Module, output: Any, *inputs: Any) -> Any:  # type: ignore[override]
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
    def backward(ctx, output_grads: Tuple[Optional[torch.Tensor]]) -> Any:  # type: ignore[override]
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
        outputs_requires_grad: List[torch.Tensor] = []
        output_grad_tensors: List[torch.Tensor] = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_requires_grad.append(outputs[i])
                assert (
                    output_grads[i] is not None
                ), f"expecting grad for output at index {i}, but got None."
                output_grad_tensors.append(output_grads[i])  # type: ignore[arg-type]
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

    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any]) -> None:
        if checkpoint.state(module).enable_hook:
            checkpoint.state(module).orig_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

    def forward_hook(module: nn.Module, inputs: Tuple[Any], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            torch.set_grad_enabled(checkpoint.state(module).orig_grad_enabled)
            return _ModuleHookCheckpointFunction.apply(module, output, *inputs)
        else:
            return output

    # This hook does the following things:
    # 1. detach outputs from the autograd graph to discard activations
    # 2. insert an autograd.Function after the forward pass to recompute
    #    activations during the backward pass.
    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook)
    # Use prepend to make sure we restore the original grad enabled state right
    # after the module forward invocation.
    module.register_forward_hook(forward_hook, prepend=True)
    return module
