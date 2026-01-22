"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import warnings
from typing import Any, TYPE_CHECKING

import torch
from torch.export.dynamic_shapes import _DimHint, Dim
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.utils import _pytree


if TYPE_CHECKING:
    from collections.abc import Sequence


def from_dynamic_axes_to_dynamic_shapes(
    model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    *,
    dynamic_axes=None,
    output_names: set[str],
    input_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any | None] | None, tuple[Any, ...], dict[str, Any] | None]:
    """
    Converts dynamic_axes into dynamic_shapes by wrapping the axis names with ``torch.export.Dim.DYNAMIC``.

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}
    (2) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}

    Detail on Dim.DYNAMIC: `#133620 <https://github.com/pytorch/pytorch/pull/133620>`_
    """

    warnings.warn(
        "from_dynamic_axes_to_dynamic_shapes is deprecated and will be removed in a future release. "
        "This function converts 'dynamic_axes' format (including custom axis names) to 'dynamic_shapes' format. "
        "Instead of relying on this conversion, provide 'dynamic_shapes' directly with custom names.",
        DeprecationWarning,
        stacklevel=2,
    )

    # https://github.com/pytorch/pytorch/pull/128371
    # 1. The function does not need to provide dynamic_shapes to torch.export.export
    if dynamic_axes is None:
        return None, args, kwargs

    if input_names is None:
        input_names = []

    if kwargs is None:
        kwargs = {}

    dynamic_shapes: dict[str, Any | None] = {}
    for input_name, axes in dynamic_axes.items():
        # NOTE: torch.export.Dim.DYNAMIC does its best to infer the min and max values
        # from the model, but it's not guaranteed to be dynamic.
        if input_name in output_names:
            # output names are not needed for dynamic_shapes
            continue
        if isinstance(axes, dict):
            if any(not isinstance(k, int) for k in axes):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            # str will be converted to Dim.DYNAMIC in convert_str_to_export_dim
            dynamic_shapes[input_name] = axes
        elif isinstance(axes, list):
            if any(not isinstance(k, int) for k in axes):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            dynamic_shapes[input_name] = dict.fromkeys(axes, torch.export.Dim.DYNAMIC)
        elif axes is None:
            dynamic_shapes[input_name] = None
        else:
            raise ValueError(
                "Unsupported dynamic_axes format. Please provide a dict or a list."
            )

    for input_name in input_names:
        if input_name not in dynamic_shapes:
            dynamic_shapes[input_name] = None

    # Order the inputs according to the signature of the model
    sig = _signature(model)
    inputs = []
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(args):
            inputs.append(args[idx])
        elif param_name in kwargs:
            inputs.append(kwargs[param_name])

    # We need tree structure to represent dynamic_shapes
    dynamic_shapes = _unflatten_dynamic_shapes_with_inputs_tree(inputs, dynamic_shapes)

    # Since the dynamic_shapes are now in the order of the model parameters,
    # we need to convert args and kwargs to the order of the model parameters.
    return dynamic_shapes, tuple(inputs), {}


def from_dynamic_shapes_to_dynamic_axes(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    input_names: Sequence[str],
    exception: Exception,
) -> dict[str, Any] | None:
    """
    Converts dynamic_shapes into dynamic_axes by removing torch.export.Dim wrapping
    and converting to list or dict form based on whether dimension names are present.

    dynamic_shapes examples:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = ({0: Dim("my_custom_axis_name_1"}, {1: Dim("my_custom_axis_name_2")})

    these will be converted to dynamic_axes respectively:
    (1) dynamic_axes = {"x": [0], "y": [1]}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    NOTE: If the model input is nested, so is the dynamic_shapes, we need to flatten the dynamic_shapes,
    and then assign the axes to the input names in the order they are provided.

    NOTE: input_names are used to assign the axes to the correct input names. If the input names are not
    provided, or less than the dynamic inputs/axes, it raises an error.
    """

    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)

    if len(input_names) < len(flat_dynamic_shapes):
        raise ValueError(
            "To construct dynamic_axes from dynamic_shapes, "
            f"number of input names ({len(input_names)}) should be greater than or equal to "
            f"the number of graph inputs(flat) ({len(flat_dynamic_shapes)})"
        ) from exception

    dynamic_axes: dict[str, list[int]] = {}
    # input names are assigned in order
    for input_name, axes in zip(input_names, flat_dynamic_shapes):
        if axes is None:
            continue

        converted_axes: list[int] = []
        if isinstance(axes, dict):
            for axis, dim in axes.items():
                if dim is None:
                    continue
                converted_axes.append(axis)
            dynamic_axes[input_name] = converted_axes
        elif isinstance(axes, (list, tuple)):
            for idx, dim in enumerate(axes):
                if dim is None:
                    continue
                converted_axes.append(idx)
            dynamic_axes[input_name] = converted_axes
    return dynamic_axes


def _any_str_or_dim_in_dynamic_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
) -> bool:
    """Check if there is any string or Dim in the dynamic_shapes."""
    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)
    # This indicates the dynamic_shapes includes something we don't support in axes, and it's flattened
    # to itself. Otherwise, flat_dynamic_shapes should be a list of dict/list/tuple (or None).
    if any(
        not isinstance(axes, (dict, list, tuple)) and axes is not None
        for axes in flat_dynamic_shapes
    ):
        return False
    # both str and Dim can provide custom names
    for axes in flat_dynamic_shapes:
        if isinstance(axes, dict):
            for dim in axes.values():
                if isinstance(dim, (str, Dim)):
                    return True
        elif isinstance(axes, (list, tuple)):
            for dim in axes:
                if isinstance(dim, (str, Dim)):
                    return True
    return False


def convert_str_to_export_dim(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
) -> tuple[dict[str, Any] | tuple[Any, ...] | list[Any] | None, bool]:
    # 1. If there is no string in dynamic_shapes, we do not touch dynamic_shapes
    if dynamic_shapes is None or not _any_str_or_dim_in_dynamic_shapes(dynamic_shapes):
        return dynamic_shapes, False
    # 2. Convert "name" to Dim.DYNAMIC with flattening and identify if there is any string
    #    to be replaced with Dim.DYNAMIC, and then unflatten it back to the original structure.
    #    for example: {"y": {0: "dim_0"}, "x": {1: "dim_1"}}
    #    to {"y": {0: Dim.DYNAMIC}, "x": {1: Dim.DYNAMIC}}
    dynamic_shapes_with_export_dim: list[
        list[Dim | _DimHint | None] | dict[int, Dim | _DimHint | None] | None
    ] = []
    flat_dynamic_shapes, tree_structure = _flatten_dynamic_shapes_to_axes(
        dynamic_shapes
    )
    for axes in flat_dynamic_shapes:
        if axes is None:
            dynamic_shapes_with_export_dim.append(None)
        elif isinstance(axes, dict):
            converted_axes_dict: dict[int, Dim | _DimHint | None] = {}
            for axis, dim in axes.items():
                if isinstance(dim, str):
                    converted_axes_dict[axis] = torch.export.Dim.DYNAMIC
                else:
                    converted_axes_dict[axis] = dim
            dynamic_shapes_with_export_dim.append(converted_axes_dict)
        elif isinstance(axes, (list, tuple)):
            converted_axes_list: list[Dim | _DimHint | None] = []
            for dim in axes:
                if isinstance(dim, str):
                    converted_axes_list.append(torch.export.Dim.DYNAMIC)
                else:
                    converted_axes_list.append(dim)
            dynamic_shapes_with_export_dim.append(converted_axes_list)

    dynamic_shapes_with_export_dim = _pytree.tree_unflatten(
        dynamic_shapes_with_export_dim, tree_structure
    )
    return (
        dynamic_shapes_with_export_dim,
        True,
    )


def create_rename_mapping(
    inputs, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]
) -> dict[str, str]:
    """Create a mapping from old names to new names for dynamic axes."""

    # NOTE: There's no need to handle cases where kwargs are out of order with the model signature,
    # as torch.export.export supports dynamism only when kwargs and dynamic_shapes are provided in order.
    # Reference: https://github.com/pytorch/pytorch/blob/49082f9dba3b79a344cb03652972ddbe7c3729cc/torch/export/_trace.py#L2034

    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)
    if len(inputs) != len(flat_dynamic_shapes):
        warnings.warn(
            "# ONNX model has different number of inputs than the flatten dynamic_shapes. "
            "The dynamic axes will not be renamed.",
            UserWarning,
            stacklevel=3,
        )
        return {}
    rename_mapping: dict[str, str] = {}
    # NOTE: We assume that the flat_dynamic_shapes is in the same order as the inputs
    # When the axis is static, or it connects to _DimHint in dynamic shapes, we skip renaming
    for idx, axes in enumerate(flat_dynamic_shapes):
        input = inputs[idx]
        if isinstance(axes, dict):
            for dim, axis in axes.items():
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                # _DimHint, int and None exists in dynamic shapes, we skip renaming
                if isinstance(axis, (_DimHint, int)) or axis is None:
                    continue
                # NOTE: ExportedProgram could give the axes the same name if they share
                # the same shape constraints.
                custom_name = _get_custom_axis_name(axis)
                if input.shape[dim].value in rename_mapping:
                    warnings.warn(
                        f"# The axis name: {custom_name} will not be used, since it shares "
                        f"the same shape constraints with another axis: {rename_mapping[input.shape[dim].value]}.",
                        stacklevel=2,
                    )
                    continue
                rename_mapping[input.shape[dim].value] = custom_name
        elif isinstance(axes, (list, tuple)):
            for dim, axis in enumerate(axes):
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                # _DimHint, int and None exists in dynamic shapes, we skip renaming
                if isinstance(axis, (_DimHint, int)) or axis is None:
                    continue
                # NOTE: ExportedProgram could give the axes the same name if they share
                # the same shape constraints.
                custom_name = _get_custom_axis_name(axis)
                if input.shape[dim].value in rename_mapping:
                    warnings.warn(
                        f"# The axis name: {custom_name} will not be used, since it shares "
                        f"the same shape constraints with another axis: {rename_mapping[input.shape[dim].value]}.",
                        UserWarning,
                        stacklevel=3,
                    )
                    continue
                rename_mapping[input.shape[dim].value] = _get_custom_axis_name(axis)
    return rename_mapping


def _get_custom_axis_name(axis: Dim | str) -> str:
    """Get the custom axis name from a torch.export.Dim."""
    if isinstance(axis, Dim):
        return axis.__name__
    return axis


def _unflatten_dynamic_shapes_with_inputs_tree(
    inputs: list[Any],
    dynamic_shapes: dict[str, Any],
) -> dict[str, Any | None]:
    _, tree_structure = _pytree.tree_flatten(inputs)
    return _pytree.tree_unflatten(dynamic_shapes.values(), tree_structure)


def _flatten_dynamic_shapes_to_axes(
    dynamic_shapes: dict[str, Any | None] | tuple[Any, ...] | list[Any],
) -> tuple[list[Any], _pytree.TreeSpec]:
    # If it's a dict/list/tuple with torch.export.Dim, we consider it's an axis to dim mapping
    def is_axes(x) -> bool:
        return (
            isinstance(x, dict)
            and all(
                isinstance(k, int)
                and (v is None or isinstance(v, (Dim, _DimHint, str, int)))
                for k, v in x.items()
            )
        ) or (
            isinstance(x, (list, tuple))
            and all(v is None or isinstance(v, (Dim, _DimHint, str, int)) for v in x)
        )

    return _pytree.tree_flatten(dynamic_shapes, is_leaf=is_axes)


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")
