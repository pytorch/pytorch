from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_map_only


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.component_benchmark.component import Component


def to_dtype(dtype: torch.dtype) -> Callable[[Component], Component]:
    """Convert floating inputs and model weights, leaving index tensors intact."""

    def transform(component: Component) -> Component:
        calls = [call.snapshot() for call in component.get_iteration_call_inputs()]
        warmup = [call.snapshot() for call in component.warmup_data]
        if not component.data_is_call_sequence:
            extra = replace(
                component, data_is_call_sequence=True
            ).get_iteration_call_inputs()[1:]
            warmup.extend(call.snapshot() for call in extra)

        def convert(value: torch.Tensor) -> torch.Tensor:
            result = value.to(dtype=dtype) if value.is_floating_point() else value
            return result.detach().requires_grad_(value.requires_grad)

        for call in [*calls, *warmup]:
            call.args, call.kwargs = tree_map_only(
                torch.Tensor, convert, (call.args, call.kwargs)
            )
            if call.output_grads is not None:
                call.output_grads = tree_map_only(
                    torch.Tensor, convert, call.output_grads
                )
        return replace(
            component,
            model=copy.deepcopy(component.model).to(dtype=dtype),
            data=calls,
            warmup_data=warmup,
        )

    return transform


def repeat_batch(
    scale_factor: int,
    tensor_batch_dim: Callable[[torch.Tensor], int | None],
) -> Callable[[Component], Component]:
    """Repeat explicit batch dimensions; return None for non-batch tensors.

    The selector must account for masks, offsets and flattened sequence data.
    Jagged containers should use a custom preprocessor with their own semantics.
    """
    if scale_factor < 1:
        raise ValueError("scale_factor must be a positive integer")

    def transform(component: Component) -> Component:
        calls = [call.snapshot() for call in component.get_iteration_call_inputs()]
        warmup = [call.snapshot() for call in component.warmup_data]
        if not component.data_is_call_sequence:
            extra = replace(
                component, data_is_call_sequence=True
            ).get_iteration_call_inputs()[1:]
            warmup.extend(call.snapshot() for call in extra)

        def repeat(value: torch.Tensor) -> torch.Tensor:
            dim = tensor_batch_dim(value)
            if dim is None:
                return value
            factors = [1] * value.ndim
            factors[dim] = scale_factor
            return value.detach().repeat(*factors).requires_grad_(value.requires_grad)

        for call in [*calls, *warmup]:
            if call.output_grads is not None:
                raise ValueError("Recapture output gradients after changing batch size")
            call.args, call.kwargs = tree_map_only(
                torch.Tensor, repeat, (call.args, call.kwargs)
            )
        return replace(component, data=calls, warmup_data=warmup)

    return transform


def compose(
    *transforms: Callable[[Component], Component],
) -> Callable[[Component], Component]:
    """Apply preprocessors in order."""

    def transform(component: Component) -> Component:
        for operation in transforms:
            component = operation(component)
        return component

    return transform
