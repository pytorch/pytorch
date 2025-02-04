"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import inspect
import re
from typing import Any, Sequence

import torch
from torch.export.dynamic_shapes import _Dim, _DimHint
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.utils import _pytree


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
    Converts dynamic_axes into dynamic_shapes by wrapping the axis names with torch.export.Dim.AUTO.

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim.AUTO}, "y": {1: Dim.AUTO}}
    (2) dynamic_shapes = {"x": {0: Dim.AUTO}, "y": {1: Dim.AUTO}}

    Detail on Dim.AUTO: https://github.com/pytorch/pytorch/pull/133620
    """
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
        # NOTE: torch.export.Dim.AUTO does its best to infer the min and max values
        # from the model, but it's not guaranteed to be dynamic.
        if input_name in output_names:
            # User specified an output name as a dynamic axis, so we skip it
            continue
        if isinstance(axes, dict):
            if any(not isinstance(k, int) for k in axes.keys()):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            dynamic_shapes[input_name] = {
                k: torch.export.Dim.AUTO for k, _ in axes.items()
            }
        elif isinstance(axes, list):
            if any(not isinstance(k, int) for k in axes):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            dynamic_shapes[input_name] = {k: torch.export.Dim.AUTO for k in axes}
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
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
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

    dynamic_axes: dict[str, list[str] | dict[int, str]] = {}
    # input names are assigned in order
    for input_name, axes in zip(input_names, flat_dynamic_shapes):
        if axes is None:
            continue
        if isinstance(axes, dict):
            converted_axes_dict: dict[int, str] = {}
            for axis, dim in axes.items():
                if dim is None:
                    continue
                # TODO(titaiwang): What if dim is DimHint?
                # can we use int?
                converted_axes_dict[axis] = dim.__name__
            dynamic_axes[input_name] = converted_axes_dict
        elif isinstance(axes, (list, tuple)):
            converted_axes_list: list[str] = []
            for dim in axes:
                if dim is None:
                    continue
                # TODO(titaiwang): What if dim is DimHint?
                # can we use int?
                converted_axes_list.append(dim.__name__)
            dynamic_axes[input_name] = converted_axes_list
    return dynamic_axes


def convert_str_to_export_dim(
    model,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
) -> dict[str, Any] | tuple[Any, ...] | list[Any] | None:
    if dynamic_shapes is None:
        return None, False
    # 1. Order inputs with model.forward signature to avoid arg mismatch
    #    for example: {"y": {0: Dim.AUTO}, "x": {1: Dim.AUTO}}
    #    Only happens when dynamic_shapes is a dict
    # NOTE: We don't support the dynamic shapes that is a nested dict with messed up order
    if isinstance(dynamic_shapes, dict):
        sig = _signature(model)
        ordered_dynamic_shapes = {}
        for param_name in sig.parameters:
            if param_name in dynamic_shapes:
                ordered_dynamic_shapes[param_name] = dynamic_shapes[param_name]
        dynamic_shapes = ordered_dynamic_shapes
    # 2. Convert "name" to Dim.AUTO with flattening and identify if there is any string
    #    to be replaced with Dim.AUTO, and then unflatten it back to the original structure.
    #    for example: {"y": {0: "dim_0"}, "x": {1: "dim_1"}}
    #    to {"y": {0: Dim.AUTO}, "x": {1: Dim.AUTO}}
    dynamic_shapes_with_export_dim: list[
        list[_Dim | _DimHint | None] | dict[int, _Dim | _DimHint | None] | None
    ] = []
    flat_dynamic_shapes, tree_structure = _flatten_dynamic_shapes_to_axes(
        dynamic_shapes
    )
    for axes in flat_dynamic_shapes:
        if axes is None:
            dynamic_shapes_with_export_dim.append(None)
        elif isinstance(axes, dict):
            converted_axes_dict: dict[int, _Dim | _DimHint | None] = {}
            for axis, dim in axes.items():
                if isinstance(dim, str):
                    converted_axes_dict[axis] = torch.export.Dim.AUTO
                else:
                    converted_axes_dict[axis] = dim
            dynamic_shapes_with_export_dim.append(converted_axes_dict)
        elif isinstance(axes, (list, tuple)):
            converted_axes_list: list[_Dim | _DimHint | None] = []
            for dim in axes:
                if isinstance(dim, str):
                    converted_axes_list.append(torch.export.Dim.AUTO)
                else:
                    converted_axes_list.append(dim)
            dynamic_shapes_with_export_dim.append(converted_axes_list)

    dynamic_shapes_with_export_dim = _pytree.tree_unflatten(
        dynamic_shapes_with_export_dim, tree_structure
    )
    return (
        dynamic_shapes_with_export_dim,
        dynamic_shapes != dynamic_shapes_with_export_dim,
    )


def create_rename_mapping(
    inputs, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]
) -> dict[str, str]:
    """Create a mapping from old names to new names for dynamic axes."""
    flat_dynamic_shapes = _flatten_dynamic_shapes_to_axes(dynamic_shapes)
    # TODO(titaiwang): Need a better error message
    assert len(inputs) == len(
        flat_dynamic_shapes
    ), "The number of ONNX graph inputs and dynamic_shapes should be the same."
    rename_mapping = {}
    # NOTE: We assume that the flat_dynamic_shapes is in the same order as the inputs
    for idx, axes in enumerate(flat_dynamic_shapes):
        input = inputs[idx]
        if isinstance(axes, dict):
            for dim, axis in axes.items():
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                rename_mapping[input.shape[dim].value] = _get_custom_axis_name(axis)
        elif isinstance(axes, (list, tuple)):
            for dim, axis in enumerate(axes):
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                rename_mapping[input.shape[dim].value] = _get_custom_axis_name(axis)
    return rename_mapping


def _get_custom_axis_name(axis: _Dim | str) -> str:
    """Get the custom axis name from a torch.export.Dim."""
    if isinstance(axis, _Dim):
        return axis.__name__
    return axis


def iterate_and_change_axis_names(
    model_or_function: ir.Model | ir.Function, rename_mapping: dict[str, str]
) -> None:
    """Rename dynamic axes in a model according to the specified dynamic_axes names."""

    for value in _all_values(model_or_function):
        if value.shape is None:
            continue
        new_shape = []
        changed = False
        for dim in value.shape:
            if not isinstance(dim, ir.SymbolicDim):
                new_shape.append(dim)
                continue
            dim_name = dim.value
            if dim_name in rename_mapping:
                new_shape.append(rename_mapping[dim_name])
                changed = True
            elif dim_name is not None:
                new_name = _replace_names(dim_name, rename_mapping)
                new_shape.append(new_name)
                if new_name != dim_name:
                    changed = True
            else:
                new_shape.append(None)
        if changed:
            value.shape = ir.Shape(new_shape)


def _unflatten_dynamic_shapes_with_inputs_tree(
    inputs: list[Any],
    dynamic_shapes: dict[str, Any],
) -> dict[str, Any | None]:
    _, tree_structure = _pytree.tree_flatten(inputs)
    return _pytree.tree_unflatten(dynamic_shapes.values(), tree_structure)


def _flatten_dynamic_shapes_to_axes(
    dynamic_shapes: dict[str, Any | None] | tuple[Any, ...] | list[Any],
) -> tuple[list[Any], _pytree.TreeSpec]:
    # If it's a dict/list/tuple with torch.export._Dim, we consider it's an axis to dim mapping
    def is_axes(x) -> bool:
        return (
            isinstance(x, dict)
            and all(
                isinstance(k, int)
                and (v is None or isinstance(v, (_Dim, _DimHint, str)))
                for k, v in x.items()
            )
            or (
                isinstance(x, (list, tuple))
                and all(isinstance(v, (_Dim, _DimHint, str)) for v in x)
            )
        )

    return _pytree.tree_flatten(dynamic_shapes, is_leaf=is_axes)


def _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:
    """Replace all known names in a shape expression with new names."""
    for old_name, new_name in rename_mapping.items():
        shape_expr = re.sub(rf"\b{old_name}\b", new_name, shape_expr)
    return shape_expr


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _all_values(model: ir.Model):
    """Yield all values in a model."""
    yield from model.graph.inputs
    yield from model.graph.initializers.values()
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        yield from node.outputs
