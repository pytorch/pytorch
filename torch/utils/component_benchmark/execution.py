from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.component_benchmark.component import (
    Component,
    copy_data,
    differentiable_tensors,
)


if TYPE_CHECKING:
    from collections.abc import Callable


class Iteration:
    """Replay a component; zero gradients and step its optimizer once per iteration."""

    def __init__(
        self,
        component: Component,
        *,
        copy_input: bool = False,
        loss_and_bwd: Callable[[Any], None] | None = None,
        torch_compile: bool = False,
        compile_options: dict[str, Any] | None = None,
        activation_checkpointing: bool = False,
        optimizer_creator_fn: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
        | None = None,
    ) -> None:
        self.component = component
        self.model = copy.deepcopy(component.model)
        self.copy_input = copy_input
        self.loss_and_bwd = loss_and_bwd
        self.checkpoint = activation_checkpointing
        self.calls = [call.snapshot() for call in component.get_iteration_call_inputs()]
        self.warmup_calls = [call.snapshot() for call in component.warmup_data]
        if not component.data_is_call_sequence:
            extra = replace(
                component, data_is_call_sequence=True
            ).get_iteration_call_inputs()[1:]
            self.warmup_calls.extend(call.snapshot() for call in extra)
        self.device = component.device
        self.optimizer = (
            optimizer_creator_fn(self.model, self.device)
            if optimizer_creator_fn
            else None
        )
        self.forward = (
            torch.compile(self.model, **(compile_options or {}))
            if torch_compile
            else self.model
        )
        self.backward = loss_and_bwd is not None or any(
            call.output_grads is not None for call in self.calls
        )
        if self.optimizer is not None and not self.backward:
            raise ValueError("An optimizer requires backward execution")
        if activation_checkpointing and not self.backward:
            raise ValueError("Activation checkpointing requires backward execution")

    def warmup(self) -> None:
        calls = self.calls
        try:
            if self.warmup_calls:
                self.calls = self.warmup_calls
                self()
        finally:
            self.calls = calls

    def __call__(self) -> list[Any]:
        self.model.zero_grad(set_to_none=True)
        outputs = []
        with torch.set_grad_enabled(self.backward):
            for call in self.calls:
                args, kwargs = (
                    copy_data((call.args, call.kwargs))
                    if self.copy_input
                    else (call.args, call.kwargs)
                )
                for tensor in differentiable_tensors((args, kwargs)):
                    tensor.grad = None
                with torch.autocast(
                    self.device.type,
                    enabled=call.autocast_enabled,
                    dtype=call.autocast_dtype,
                ):
                    if self.checkpoint:
                        output = checkpoint(
                            self.forward, *args, use_reentrant=False, **kwargs
                        )
                    else:
                        output = self.forward(*args, **kwargs)
                outputs.append(output)
                if call.output_grads is not None:
                    tensors = differentiable_tensors(output)
                    if len(tensors) != len(call.output_grads):
                        raise ValueError(
                            "Output structure changed since gradient capture"
                        )
                    pairs = [
                        (tensor, grad)
                        for tensor, grad in zip(tensors, call.output_grads)
                        if grad is not None
                    ]
                    if pairs:
                        torch.autograd.backward(
                            [tensor for tensor, _ in pairs], [grad for _, grad in pairs]
                        )
                elif self.loss_and_bwd is not None:
                    self.loss_and_bwd(output)
        if self.optimizer is not None:
            self.optimizer.step()
        return outputs
