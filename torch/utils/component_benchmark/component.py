from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils._pytree import tree_leaves, tree_map_only


if TYPE_CHECKING:
    from pathlib import Path


def copy_data(data: Any) -> Any:
    """Snapshot a pytree, preserving tensor strides, shared storage and aliases.

    Tensor leaves are detached from the parent autograd graph while retaining
    their ``requires_grad`` flags. Custom containers must be pytree registered.
    """
    memo: dict[int, torch.Tensor] = {}

    def detach(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) not in memo:
            memo[id(tensor)] = tensor.detach().requires_grad_(tensor.requires_grad)
        return memo[id(tensor)]

    return copy.deepcopy(tree_map_only(torch.Tensor, detach, data))


def differentiable_tensors(data: Any) -> list[torch.Tensor]:
    tensors: dict[int, torch.Tensor] = {}
    for leaf in tree_leaves(data):
        if isinstance(leaf, torch.Tensor) and leaf.requires_grad:
            tensors[id(leaf)] = leaf
    return list(tensors.values())


def component_loss_and_bwd(output: Any) -> None:
    """Backpropagate unit gradients through all differentiable output leaves."""
    outputs = differentiable_tensors(output)
    if not outputs:
        raise ValueError("The component has no differentiable outputs")
    torch.autograd.backward(outputs, [torch.ones_like(tensor) for tensor in outputs])


def default_loss_and_bwd(output: Any) -> None:
    """Backpropagate through the first differentiable output leaf."""
    outputs = differentiable_tensors(output)
    if not outputs:
        raise ValueError("The component has no differentiable outputs")
    outputs[0].sum().backward()


@dataclass
class CallInput:
    """One module invocation, including its effective autocast context."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None
    output_grads: list[torch.Tensor | None] | None = None

    def snapshot(self) -> CallInput:
        args, kwargs, grads = copy_data((self.args, self.kwargs, self.output_grads))
        return replace(self, args=args, kwargs=kwargs, output_grads=grads)


@dataclass
class Component:
    """A module and representative inputs for independent replay.

    With ``data_is_call_sequence=True``, one iteration replays every call in
    order. Otherwise the first input is measured and other batches warm up
    additional shapes. Data may contain ``CallInput`` objects or the tuple
    input form selected by ``data_is_args_kwargs_tuple``.
    """

    model: nn.Module
    data: list[Any]
    name: str = "model"
    data_is_args_kwargs_tuple: bool = False
    data_is_call_sequence: bool = False
    warmup_data: list[CallInput] = field(default_factory=list)

    def get_iteration_call_inputs(self) -> list[CallInput]:
        if not self.data:
            raise ValueError(f"Component {self.name!r} has no input data")
        data = self.data if self.data_is_call_sequence else self.data[:1]
        calls = []
        for value in data:
            if isinstance(value, CallInput):
                calls.append(value)
            elif self.data_is_args_kwargs_tuple:
                args, kwargs = value
                calls.append(CallInput(args=tuple(args), kwargs=dict(kwargs)))
            else:
                args = value if isinstance(value, tuple) else (value,)
                calls.append(CallInput(args=args))
        return calls

    def get_iteration_call_count(self) -> int:
        return len(self.get_iteration_call_inputs())

    @property
    def device(self) -> torch.device:
        devices = {
            value.device
            for value in [
                *self.model.parameters(),
                *self.model.buffers(),
                *tree_leaves(
                    [
                        (call.args, call.kwargs)
                        for call in self.get_iteration_call_inputs()
                    ]
                ),
            ]
            if isinstance(value, torch.Tensor)
        }
        if len(devices) > 1:
            raise ValueError(f"A component must be on one device; found {devices}")
        return next(iter(devices), torch.device("cpu"))

    def save(self, path: str | Path) -> None:
        """Save a component for local replay, including its Python module."""
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = None) -> Component:
        """Load a trusted component file. Never load an untrusted pickle."""
        value = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(value, cls):
            raise TypeError(f"Expected a Component, got {type(value).__name__}")
        return value
