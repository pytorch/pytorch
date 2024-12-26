"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnxscript_apis, onnxscript_ir as ir
from torch.onnx._internal.exporter import _core, _onnx_program, _registration
from torch.utils import _pytree


if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _rename_dynamic_shapes_with_model_inputs(
    model,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any],
    input_names: Sequence[str],
) -> dict[str, Any] | tuple[Any] | list[Any]:
    """

    This function renames the dynamic_shapes with the parameters of the model, since
    torch.export.export requires the dynamic_shapes to be named with the model's input names.

    NOTE: If the model input is nested, this function does nothing, and the users are responsible
    for providing the correct dynamic_shapes with the correct model parameters as keys. However,
    dynamic_shapes is usually defined as a tuple when the input is nested.

    """
    if isinstance(dynamic_shapes, (tuple, list)):
        # It doesn not specify input names if it's a tuple
        return dynamic_shapes

    sig = _signature(model)

    # This indicates that inputs are nested, and users specify
    # flattened input names, so we don't rename accordingly.
    # If users really assign customized names to the nested inputs, they
    # get errors from torch.export.export
    if len(input_names) != len(sig.parameters):
        return dynamic_shapes

    renamed_dynamic_shapes = {}
    for idx, param_name in enumerate(sig.parameters):
        input_name = input_names[idx]
        if input_name in dynamic_shapes:
            renamed_dynamic_shapes[param_name] = dynamic_shapes[input_name]

    return renamed_dynamic_shapes


def _from_dynamic_axes_to_dynamic_shapes(
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
        # TODO(titaiwang): Add ONNX IR pass to rename default dynamic axes: s0, s1, ...
        # to the dynamic axes defined by users.
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


def _unflatten_dynamic_shapes_with_inputs_tree(
    inputs: list[Any],
    dynamic_shapes: dict[str, Any | None],
) -> dict[str, Any | None]:
    _, tree_structure = _pytree.tree_flatten(inputs)
    return _pytree.tree_unflatten(dynamic_shapes.values(), tree_structure)


def _from_dynamic_shapes_to_dynamic_axes(
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

    # 0. flatten the dynamic_shapes
    # If it's a dict with torch.export._Dim, we consider it's an axis to dim mapping
    def is_dict_axes(x) -> bool:
        # TODO: torch.export._Dim is not exposed, so we use a hacky way to check the type
        return isinstance(x, dict) and all(
            isinstance(k, int)
            and (v is None or isinstance(v, torch.export.Dim("test").__class__))
            for k, v in x.items()
        )

    flat_dynamic_shapes = _pytree.tree_leaves(dynamic_shapes, is_leaf=is_dict_axes)

    if len(input_names) < len(flat_dynamic_shapes):
        raise ValueError(
            "To construct dynamic_axes from dynamic_shapes, "
            f"number of input names ({len(input_names)}) should be greater than or equal to "
            f"the number of graph inputs(flat) ({len(flat_dynamic_shapes)})"
        ) from exception

    dynamic_axes = {}
    # input names are assigned in order
    for input_name, axes in zip(input_names, flat_dynamic_shapes):
        if axes is None:
            continue
        converted_axes = {}
        for axis, dim in axes.items():
            if dim is None:
                continue
            converted_axes[axis] = dim.__name__
            dynamic_axes[input_name] = converted_axes
    return dynamic_axes


def _get_torch_export_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    return args, kwargs


def export_compat(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...],
    f: str | os.PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    keep_initializers_as_inputs: bool = False,
    external_data: bool = True,
    report: bool = False,
    optimize: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    fallback: bool = False,
    **_,
) -> _onnx_program.ONNXProgram:
    if opset_version is None:
        opset_version = onnxscript_apis.torchlib_opset_version()

    if isinstance(model, torch.export.ExportedProgram):
        # We know the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used
        dynamic_shapes = dynamic_shapes or {}
    else:
        args, kwargs = _get_torch_export_args(args, kwargs)
        if dynamic_shapes is None and dynamic_axes is not None:
            warnings.warn(
                "# 'dynamic_axes' is not recommended when dynamo=True, "
                "and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' "
                "Supply the 'dynamic_shapes' argument instead if export is unsuccessful.",
                UserWarning,
            )
            try:
                dynamic_shapes, args, kwargs = _from_dynamic_axes_to_dynamic_shapes(
                    model,
                    args,
                    kwargs,
                    dynamic_axes=dynamic_axes,
                    input_names=input_names,
                    output_names=set(output_names or ()),
                )
            except Exception as e:
                raise RuntimeError(
                    "# Failed to convert 'dynamic_axes' to 'dynamic_shapes'. "
                    "Please provide 'dynamic_shapes' directly. "
                    "Refer to the documentation for 'torch.export.export' for more information on dynamic shapes."
                ) from e
        elif dynamic_shapes is not None and input_names is not None:
            # NOTE: If dynamic_shapes and input_names are both provided, we need to check
            # if dynamic_shapes is using input_names. If so, we need to internally change it to
            # model inputs to be compatible with torch.export.export
            dynamic_shapes = _rename_dynamic_shapes_with_model_inputs(
                model,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names,
            )

    registry = _registration.ONNXRegistry.from_torchlib()
    if custom_translation_table is not None:
        for torch_op, onnx_ops in custom_translation_table.items():
            # TODO(justinchuby): Support complex inputs with annotations
            if not isinstance(onnx_ops, Sequence):
                onnx_ops = (onnx_ops,)
            for op in reversed(onnx_ops):
                # register_op places the op in the front of all onnx variants,
                # so we reverse the list to maintain the order of the custom ops provided
                registry.register_op(torch_op, op, is_complex=False)
    try:
        onnx_program = _core.export(
            model,
            args,
            kwargs,
            registry=registry,
            dynamic_shapes=dynamic_shapes,
            input_names=input_names,
            output_names=output_names,
            profile=profile,
            report=report,
            verify=verify,
            dump_exported_program=dump_exported_program,
            artifacts_dir=artifacts_dir,
            verbose=verbose,
        )

    except Exception as e:
        if fallback:
            if verbose is not False:
                print(
                    "[torch.onnx] Falling back to legacy torch.onnx.export due "
                    f"to the following error: {e}",
                )
            if f is None:
                raise TypeError("f must be provided when fallback is enabled") from e
            if dynamic_shapes is not None and dynamic_axes is None:
                if input_names is None:
                    raise ValueError(
                        "Failed to convert dynamic_shapes to dynamic_axes. "
                        "Either input_names or dynamic_axes must be provided "
                        "when dynamic is requested in fallback"
                    ) from e
                dynamic_axes = _from_dynamic_shapes_to_dynamic_axes(
                    dynamic_shapes=dynamic_shapes, input_names=input_names, exception=e
                )
            torch.onnx.utils.export(
                model,  # type: ignore[arg-type]
                args,
                f,  # type: ignore[arg-type]
                kwargs=kwargs,
                export_params=export_params,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,  # TODO(justinchuby): Hard coded to 17 for now
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
            )
            onnx_program = _onnx_program.ONNXProgram(ir.load(f), None)

            # NOTE: It it's falling back to the legacy exporter, we don't need to
            # optimize the model, so we return it here. Users can still optimize
            # the model using the optimize() if they want.
            return onnx_program
        else:
            raise

    # Converter opset version and optimize
    onnx_program.model = onnxscript_apis.convert_version(
        onnx_program.model, opset_version
    )
    if optimize:
        onnx_program.optimize()

    if f is not None:
        onnx_program.save(
            f,
            include_initializers=export_params,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            external_data=external_data,
        )

    return onnx_program
