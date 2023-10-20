"""Module for handling op-level validation during exporting."""

from __future__ import annotations

import logging

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]

import torch
import torch.fx

from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    diagnostics,
    fx_onnx_interpreter,
    type_utils as fx_type_utils,
)
from torch.utils import _pytree


@_beartype.beartype
def _op_level_debug_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    *args,
    **kwargs,
) -> str:
    return (
        f"FX Node: {node.op}::{node.target}[name={node.name}]. \n"
        f"ONNX Node: {symbolic_fn.name}[opset={symbolic_fn.opset}]."
    )


@_beartype.beartype
@diagnostics.diagnose_call(
    diagnostics.rules.op_level_debugging,
    diagnostic_message_formatter=_op_level_debug_message_formatter,
)
def validate_op_between_ort_torch(
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    fx_args: List[fx_type_utils.Argument],
    fx_kwargs: Dict[str, fx_type_utils.Argument],
    fx_graph_module: torch.fx.GraphModule,
):
    """Validate the op between ONNX Runtime and PyTorch.

    The function will run the op in ONNX Runtime and PyTorch and compare the
    results. It doesn't break the exporting process, but saves each op validated
    result into SARIF, under the section of `fx_onnx_interpreter`.

    There are three signs can be found:
    1. Blue: Pass
    2. Yellow: Bypass

    Args:
        node (torch.fx.Node): The validated fx.node
        symbolic_fn (Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]): The corresponded ONNX node
        torch_args (list): torch argument inputs
        torch_kwargs (dict): torch keyword argument inputs
        fx_graph_module (torch.fx.GraphModule): The fx.GraphModule that contains the nodes
    """
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)

    try:
        torch_args, torch_kwargs = _wrap_fx_args_as_torch_args(
            fx_args, fx_kwargs, fx_graph_module
        )
    except ValueError as value_error:
        diagnostic = diagnostic_context.inflight_diagnostic()
        with diagnostic.log_section(
            logging.WARNING, "Op level debug fails due to unsupported input types"
        ):
            diagnostic.log_source_exception(logging.WARNING, value_error)
        diagnostic.level = diagnostics.levels.WARNING
        return

    with evaluator.default_as(evaluator.ort_evaluator):
        try:
            expected_outputs = node.target(*torch_args, **torch_kwargs)  # type: ignore[operator]
        # NOTE: randomly generating indices/dim: INT64 could go out of bounds
        except IndexError as index_error:
            # TODO(titaiwang): How to bound indices/dim: INT64
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, "Op level debug is bypassed"):
                diagnostic.log_source_exception(logging.WARNING, index_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        # NOTE: Error in torch ops with random inputs generated from FakTensors
        except RuntimeError as runtime_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(
                logging.WARNING, "Op level debug fails on PyTorch"
            ):
                diagnostic.log_source_exception(logging.WARNING, runtime_error)
            diagnostic.level = diagnostics.levels.WARNING
            return

        try:
            (
                function_eager_inputs,
                function_eager_attributes,
            ) = _convert_torch_args_to_onnxfunction_args(
                symbolic_fn.param_schemas(),
                torch_args,
                torch_kwargs,
                allow_extra_kwargs=True,
            )
            # NOTE: Apply kwargs preprocessing AFTER they are split
            function_eager_attributes = (
                fx_onnx_interpreter.filter_incompatible_and_dtype_convert_kwargs(
                    function_eager_attributes
                )
            )
        # NOTE: Incompatible kwargs or missing required args
        except TypeError as type_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, "Op level debug is bypassed"):
                diagnostic.log_source_exception(logging.WARNING, type_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        try:
            ort_outputs = symbolic_fn(
                *function_eager_inputs, **function_eager_attributes
            )
        # NOTE: Error in ONNX Runtime with random inputs generated from FakTensors
        except RuntimeError as runtime_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(
                logging.WARNING, "Op level debug fails on ONNXRUNTIME"
            ):
                diagnostic.log_source_exception(logging.WARNING, runtime_error)
            diagnostic.level = diagnostics.levels.WARNING
            return

        flattened_torch_outputs, _ = _pytree.tree_flatten(expected_outputs)
        flattened_function_outputs, _ = _pytree.tree_flatten(ort_outputs)

        assert flattened_torch_outputs
        assert len(flattened_torch_outputs) == len(flattened_function_outputs)

        for torch_output, function_output in zip(
            flattened_torch_outputs, flattened_function_outputs
        ):
            try:
                if isinstance(function_output, onnxscript.tensor.Tensor):
                    function_output = function_output.value

                # Use torch.testing as opposed to np.testing to ensure dtypes and shapes match
                torch.testing.assert_close(
                    torch.tensor(function_output).cpu(),
                    torch_output.cpu()
                    if isinstance(torch_output, torch.Tensor)
                    else torch.tensor(torch_output).cpu(),
                    rtol=1e-4,
                    atol=1e-3,
                )
            except AssertionError as e:
                diagnostic = diagnostic_context.inflight_diagnostic()
                with diagnostic.log_section(logging.WARNING, "Validation failed"):
                    diagnostic.log_source_exception(logging.WARNING, e)
                diagnostic.level = diagnostics.levels.WARNING


@_beartype.beartype
def _convert_symint_to_int_in_shape(shape: torch.Size) -> torch.Size:
    """Convert SymInt to int in shape

    Args:
        shape (torch.Size): The shape of a tensor
    Raises:
        ValueError: When SymInt is found in shape
    Returns:
        torch.Size: The shape of a tensor with SymInt converted to int

    """
    list_int_shape = []
    for dim in shape:
        if isinstance(dim, torch.SymInt):
            if symbolic_shapes.has_hint(dim):
                list_int_shape.append(symbolic_shapes.hint_int(dim))
            else:
                raise ValueError(
                    f"An unbacked SymInt found in shape. SymInt: {dim}; "
                    f"torch.Size: {shape}. There is no hint for SymInt."
                )
        else:
            list_int_shape.append(dim)
    return torch.Size(list_int_shape)


@_beartype.beartype
def generate_random_tensors(shape: torch.Size, dtype: torch.dtype):
    shape = _convert_symint_to_int_in_shape(shape)

    if dtype == torch.uint8:
        return torch.randint(
            low=_constants.UINT8_MIN, high=_constants.UINT8_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int8:
        return torch.randint(
            low=_constants.INT8_MIN, high=_constants.INT8_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int16:
        return torch.randint(
            low=_constants.INT16_MIN, high=_constants.INT16_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int32:
        return torch.randint(
            low=_constants.INT32_MIN, high=_constants.INT32_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int64:
        return torch.randint(
            low=_constants.INT64_MIN, high=_constants.INT64_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.bool:
        random_numbers = torch.rand(shape)
        return torch.where(
            random_numbers > 0.5, torch.tensor(True), torch.tensor(False)
        )
    if fx_type_utils.is_torch_complex_dtype(dtype):
        # ONNX does not support complex values, but supports their real representation
        return torch.randn(
            (*shape, 2), dtype=fx_type_utils.from_complex_to_float(dtype)
        )
    return torch.randn(shape, dtype=dtype)


@_beartype.beartype
def _fx_args_to_torch_args(
    fx_args: List[fx_type_utils.Argument], fx_graph_module: torch.fx.GraphModule
) -> List[fx_type_utils.Argument]:
    """Recursively convert fx args to torch args"""
    wrapped_args: List[fx_type_utils.Argument] = []
    for arg in fx_args:
        if isinstance(arg, torch.fx.Node):
            fake_tensor = arg.meta.get("val")
            if fake_tensor is None and arg.op == "get_attr":
                fake_tensor = getattr(fx_graph_module, arg.target)  # type: ignore[operator]
            # NOTE: Currently, we are aware of
            # FakeTensor/Tensor/SymInt/SymFloat/Symbool/int/float/bool could be in
            # arg.meta["val"]/get_attr.
            if isinstance(fake_tensor, torch.Tensor):
                real_tensor = generate_random_tensors(
                    fake_tensor.shape, fake_tensor.dtype
                )
                wrapped_args.append(real_tensor)
            elif isinstance(fake_tensor, (int, float, bool)):
                wrapped_args.append(fake_tensor)
            elif symbolic_shapes.has_hint(fake_tensor):
                wrapped_args.append(symbolic_shapes.hint_int(fake_tensor))
            else:
                raise ValueError(
                    f"Unexpected input argument type found inside fx.Node. arg: {arg}; "
                    f"arg.meta['val']/get_attr: {fake_tensor}; type(arg.meta['val']/get_attr): "
                    f"{type(fake_tensor)}."
                )
        elif isinstance(arg, Sequence):
            wrapped_args.append(_fx_args_to_torch_args(arg, fx_graph_module))
        elif isinstance(arg, (int, float, torch.dtype)) or arg is None:
            wrapped_args.append(arg)
        elif isinstance(arg, torch.device):
            wrapped_args.append(str(arg))
        else:
            raise ValueError(
                f"Unexpected input argument type is found in node arguments. arg: {arg}; "
            )

    return wrapped_args


@_beartype.beartype
def _wrap_fx_args_as_torch_args(
    fx_args: List[fx_type_utils.Argument],
    fx_kwargs: Dict[str, fx_type_utils.Argument],
    fx_graph_module: torch.fx.GraphModule,
) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Prepare torch format args and kwargs for op-level validation by using fake tensor to create real tensor to feed in ops"""

    # NOTE: This function only supports FakeTensor with concrete shapes
    torch_args: List[fx_type_utils.Argument] = _fx_args_to_torch_args(
        fx_args, fx_graph_module
    )
    return torch_args, fx_kwargs


# NOTE: Referenced from onnxscript internal function: _tag_arguments_with_param_schemas.
@_beartype.beartype
def _convert_torch_args_to_onnxfunction_args(
    param_schemas: Sequence[onnxscript.values.ParamSchema],
    args: List[fx_type_utils.Argument],
    kwargs: Dict[str, fx_type_utils.Argument],
    allow_extra_kwargs: bool = False,
) -> Tuple[List[Any], Dict[str, Any],]:
    """Convert Python args and kwargs to OnnxFunction acceptable with matching ONNX ParamSchema.

    NOTE: This is different from the param_schema separating in dispatcher, since at this point
    we are already sure that the args and kwargs are in order and matched.

    Args:
        param_schemas: The parameter schemas of an Op or a OnnxFunction.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.
        allow_extra_kwargs: Whether to allow extra keyword arguments.
            When set to True, extra/unknown arguments will be ignored.

    Returns:
        A tuple of two elements:
        - A list of Python positional argument.
        - An ordered dictionary of Python keyword argument names and its values.

    Raises:
        TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
        TypeError: When a required input is not provided.
    """
    # args, kwargs and param_schemas should be all in order
    # user may not specify all inputs or attributes

    all_param_names = {param.name for param in param_schemas}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    tagged_args: list[Any] = []
    tagged_kwargs: dict[str, Any] = {}

    for i, param in enumerate(param_schemas):
        if param.is_variadic_input:
            # Exhaust all remaining args
            tagged_args.extend(arg for arg in args[i:])
            args = []
            continue
        if i < len(args):
            if param.is_input or isinstance(args[i], torch.dtype):
                tagged_args.append(_convert_tensor_to_numpy(args[i]))
            else:
                tagged_args.append(args[i])
        elif param.name in kwargs:
            if param.is_input:
                tagged_kwargs[param.name] = _convert_tensor_to_numpy(kwargs[param.name])
            else:
                tagged_kwargs[param.name] = kwargs[param.name]
        elif param.required:
            raise TypeError(f"Required input/attribute '{param}' was not provided")

    return tagged_args, tagged_kwargs


@_beartype.beartype
def _convert_tensor_to_numpy(input: fx_type_utils.Argument) -> Any:
    try:
        import numpy as np
    except ImportError:
        raise ImportError(f"{__name__} needs numpy, but it's not installed.")

    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    if isinstance(input, torch.dtype):
        return int(jit_type_utils.JitScalarType.from_dtype(input).onnx_type())  # type: ignore[union-attr]
    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if isinstance(input[0], torch.Tensor):
            return [_convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)
    return input
