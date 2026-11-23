from __future__ import annotations

import copy
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.component_benchmark.component import (
    CallInput,
    Component,
    component_loss_and_bwd,
    copy_data,
    differentiable_tensors,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def get_submodules_with_depths(
    root_module: nn.Module, max_depth: int, root_module_path: str = ""
) -> list[str]:
    """List module paths in breadth-first order, including the root."""
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    queue = deque([(root_module, root_module_path, 0)])
    paths = []
    while queue:
        module, path, depth = queue.popleft()
        paths.append(path)
        if depth < max_depth:
            for name, child in module.named_children():
                queue.append((child, f"{path}.{name}" if path else name, depth + 1))
    return paths


def _capture_gradients(call: CallInput, output: Any, stack: ExitStack) -> None:
    outputs = differentiable_tensors(output)
    call.output_grads = [None] * len(outputs)

    def make_hook(index: int) -> Callable[[torch.Tensor], None]:
        def save(grad: torch.Tensor) -> None:
            if call.output_grads is not None:
                call.output_grads[index] = grad.detach().clone()

        return save

    for index, tensor in enumerate(outputs):
        stack.callback(tensor.register_hook(make_hook(index)).remove)


def capture_components(
    component: Component,
    paths: list[str],
    *,
    capture_output_grad: bool = False,
    loss_and_bwd: Callable[[Any], None] = component_loss_and_bwd,
) -> dict[str, Component]:
    """Capture all invocations during one parent iteration.

    The parent model and inputs are copied before execution. Hooks, autograd
    graphs and RNG changes are confined to this capture. An inactive module
    has an empty call sequence; it must not be measured as a successful run.
    """
    model = copy.deepcopy(component.model)
    calls: dict[str, list[CallInput]] = {path: [] for path in paths}
    pending: dict[str, list[CallInput]] = {path: [] for path in paths}
    initial_models = {path: component.model.get_submodule(path) for path in paths}
    device = component.device
    cuda_devices = [device.index] if device.type == "cuda" else []

    def make_pre_hook(path: str) -> Callable[..., None]:
        def capture(
            module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            saved_args, saved_kwargs = copy_data((args, kwargs))
            enabled = torch.is_autocast_enabled(device.type)
            call = CallInput(
                args=saved_args,
                kwargs=saved_kwargs,
                autocast_enabled=enabled,
                autocast_dtype=torch.get_autocast_dtype(device.type)
                if enabled
                else None,
            )
            calls[path].append(call)
            pending[path].append(call)

        return capture

    def make_post_hook(path: str, stack: ExitStack) -> Callable[..., None]:
        def capture(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            call = pending[path].pop()
            if capture_output_grad:
                _capture_gradients(call, output, stack)

        return capture

    with ExitStack() as stack, torch.random.fork_rng(devices=cuda_devices):
        for path in paths:
            module = model.get_submodule(path)
            stack.callback(
                module.register_forward_pre_hook(
                    make_pre_hook(path), with_kwargs=True
                ).remove
            )
            stack.callback(
                module.register_forward_hook(make_post_hook(path, stack)).remove
            )
        batches = (
            [component.get_iteration_call_inputs()]
            if component.data_is_call_sequence
            else [
                replace(component, data=[value]).get_iteration_call_inputs()
                for value in component.data
            ]
        )
        first_batch_counts = dict.fromkeys(paths, 0)
        with torch.set_grad_enabled(capture_output_grad or torch.is_grad_enabled()):
            for index, batch in enumerate(batches):
                for call in batch:
                    args, kwargs = copy_data((call.args, call.kwargs))
                    with torch.autocast(
                        device.type,
                        enabled=call.autocast_enabled,
                        dtype=call.autocast_dtype,
                    ):
                        output = model(*args, **kwargs)
                    if capture_output_grad:
                        loss_and_bwd(output)
                if index == 0:
                    first_batch_counts = {
                        path: len(data) for path, data in calls.items()
                    }

    return {
        path: replace(
            component,
            model=initial_models[path],
            data=calls[path][: first_batch_counts[path]],
            warmup_data=calls[path][first_batch_counts[path] :],
            name=f"{component.name}.{path}" if path else component.name,
            data_is_call_sequence=True,
            data_is_args_kwargs_tuple=False,
        )
        for path in paths
    }
