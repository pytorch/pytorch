# mypy: allow-untyped-defs
from __future__ import annotations


"""Model summary utilities for inspecting nn.Module architectures.

This module provides a Keras-style summary function that displays layer names,
output shapes, and parameter counts for PyTorch models.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch import nn, Tensor


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["summary", "ModelSummary", "LayerInfo"]


@dataclass
class LayerInfo:
    """Information about a single layer in the model.

    Attributes:
        name: Fully qualified name of the module (e.g., "layer1.conv").
        class_name: Class name of the module (e.g., "Conv2d").
        depth: Nesting depth in the module hierarchy (0 = root).
        output_size: Shape of the output tensor from this layer.
        num_params: Total number of parameters in this layer (non-recursive).
        trainable_params: Number of trainable parameters in this layer.
    """

    name: str
    class_name: str
    depth: int
    output_size: tuple[int, ...]
    num_params: int
    trainable_params: int


class ModelSummary:
    """Container for model summary information.

    This class holds the summary data for a model and provides formatted
    string output when printed.

    Attributes:
        model_name: Name of the model class.
        layer_info: List of LayerInfo objects for each module.
        total_params: Total number of parameters in the model.
        trainable_params: Number of trainable parameters.
        max_depth: Maximum depth of modules to display.
    """

    def __init__(
        self,
        model_name: str,
        layer_info: list[LayerInfo],
        total_params: int,
        trainable_params: int,
        max_depth: int,
    ) -> None:
        self.model_name = model_name
        self.layer_info = layer_info
        self.total_params = total_params
        self.trainable_params = trainable_params
        self.max_depth = max_depth

    def __repr__(self) -> str:
        return self._format_table()

    def _format_table(self) -> str:
        """Format summary as ASCII table."""
        lines = []
        sep = "=" * 80

        lines.append(sep)
        lines.append(f"{'Layer (type)':<45}{'Output Shape':<20}{'Param #':>15}")
        lines.append(sep)

        for info in self.layer_info:
            if self.max_depth >= 0 and info.depth > self.max_depth:
                continue

            # Build indented name with tree characters
            if info.depth == 0:
                indent = ""
                prefix = ""
            else:
                indent = "│   " * (info.depth - 1)
                prefix = "├─"

            # Use short name (last part of FQN) for display
            short_name = info.name.split(".")[-1] if info.name else "root"
            name_str = f"{indent}{prefix}{info.class_name}: {short_name}"

            # Format output shape
            if info.output_size:
                shape_str = str(list(info.output_size))
            else:
                shape_str = "--"

            # Format param count
            if info.num_params > 0:
                param_str = f"{info.num_params:,}"
            else:
                param_str = "--"

            lines.append(f"{name_str:<45}{shape_str:<20}{param_str:>15}")

        lines.append(sep)
        lines.append(f"Total params: {self.total_params:,}")
        lines.append(f"Trainable params: {self.trainable_params:,}")
        non_trainable = self.total_params - self.trainable_params
        lines.append(f"Non-trainable params: {non_trainable:,}")
        lines.append(sep)

        return "\n".join(lines)


def _get_output_shape(output: Any) -> tuple[int, ...]:
    """Extract shape from forward output.

    Args:
        output: Output from a module's forward pass. Can be a Tensor,
            tuple/list of Tensors, or other types.

    Returns:
        Shape tuple of the output tensor, or empty tuple if no tensor found.
    """
    if isinstance(output, Tensor):
        return tuple(output.shape)
    if isinstance(output, (tuple, list)) and len(output) > 0:
        # Return shape of first tensor in sequence
        for item in output:
            if isinstance(item, Tensor):
                return tuple(item.shape)
    return ()


def summary(
    model: nn.Module,
    input_size: tuple[int, ...] | None = None,
    input_data: Tensor | None = None,
    *,
    depth: int = 3,
) -> ModelSummary:
    r"""Generate a summary of a PyTorch model.

    This function provides a Keras-style summary of the model architecture,
    showing each layer's name, output shape, and parameter count.

    Args:
        model: The nn.Module to summarize.
        input_size: Shape of input tensor, excluding batch dimension.
            For example, ``(3, 224, 224)`` for an RGB image. A batch
            dimension of 1 is automatically prepended.
        input_data: Actual input tensor to use for the forward pass.
            Alternative to ``input_size``. If provided, ``input_size``
            is ignored.
        depth: Maximum depth of nested modules to display. Default: 3.
            Use -1 for unlimited depth.

    Returns:
        ModelSummary object containing the summary information.
        The summary is also printed to stdout.

    Raises:
        ValueError: If neither ``input_size`` nor ``input_data`` is provided.

    Example::

        >>> import torch.nn as nn
        >>> from torch.nn.utils import summary
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10),
        ... )
        >>> s = summary(model, input_size=(784,))  # doctest: +SKIP
        ================================================================================
        Layer (type)                                 Output Shape        Param #
        ================================================================================
        Sequential: root                             [1, 10]                  --
        ├─Linear: 0                                  [1, 256]            200,960
        ├─ReLU: 1                                    [1, 256]                 --
        ├─Linear: 2                                  [1, 10]               2,570
        ================================================================================
        Total params: 203,530
        Trainable params: 203,530
        Non-trainable params: 0
        ================================================================================
    """
    # Validate inputs
    if input_size is None and input_data is None:
        raise ValueError(
            "Either 'input_size' or 'input_data' must be provided.\n"
            "Example: summary(model, input_size=(3, 224, 224))"
        )

    # Determine device from model
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    except StopIteration:
        # Model has no parameters, use defaults
        device = torch.device("cpu")
        dtype = torch.float32

    # Create input tensor if needed
    if input_data is None:
        assert input_size is not None  # for type checker
        # Add batch dimension
        input_shape = (1,) + input_size
        input_data = torch.zeros(input_shape, device=device, dtype=dtype)

    # Collect layer info via hooks
    layer_info_dict: dict[str, LayerInfo] = {}
    hooks: list[RemovableHandle] = []

    def make_hook(name: str, mod: nn.Module):
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            # Count parameters for this module only (not children)
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            layer_info_dict[name] = LayerInfo(
                name=name,
                class_name=module.__class__.__name__,
                depth=name.count(".") + 1 if name else 0,
                output_size=_get_output_shape(out),
                num_params=num_params,
                trainable_params=trainable,
            )

        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        hook = make_hook(name, module)
        handle = module.register_forward_hook(hook)
        hooks.append(handle)

    # Run forward pass
    original_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_data)
    finally:
        # Always cleanup: restore mode and remove hooks
        model.train(original_mode)
        for handle in hooks:
            handle.remove()

    # Calculate totals
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create summary object
    # layer_info_dict preserves insertion order (Python 3.7+)
    layer_info_list = list(layer_info_dict.values())

    summary_obj = ModelSummary(
        model_name=model.__class__.__name__,
        layer_info=layer_info_list,
        total_params=total_params,
        trainable_params=trainable_params,
        max_depth=depth,
    )

    # Print summary
    print(summary_obj)

    return summary_obj
