"""Module for handling op-level validation during exporting."""

from __future__ import annotations

import warnings

from typing import Dict, List, Sequence, Tuple, Union

import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]

import torch
import torch.fx
from torch.onnx import _constants
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.onnx._internal.fx import (
    diagnostics,
    fx_onnx_interpreter,
    type_utils as fx_type_utils,
)
from torch.utils import _pytree


@_beartype.beartype
def validate_op_between_ort_torch(
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    torch_args: tuple,
    torch_kwargs: dict,
):
    """Validate the op between ONNX Runtime and PyTorch.

    The function will run the op in ONNX Runtime and PyTorch and compare the
    results. It doesn't break the exporting process, but saves each op validated
    result into SARIF, under the section of `fx_onnx_interpreter`.

    There are three signs can be found:
    1. Blue: Pass
    2. Yellow: Bypass
    3. Red: Fail

    Args:
        node (torch.fx.Node): The validated fx.node
        symbolic_fn (Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]): The corresponded ONNX node
        torch_args (tuple): torch argument inputs
        torch_kwargs (dict): torch keyword argument inputs
    """
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)
    function_name = symbolic_fn.name

    # TODO(bowbao, titaiwang): Diagnostics.
    # - Add dedicated diagnostic for op-level validation.
    # - Consider follow up steps. E.g., dump repro.
    #   - What to do next when validation fails?
    #   - What can diagnostics offer?
    # - Warning vs Error. Should this raise?
    # - False positives. E.g., Mismatch caused by invalid random data.
    with evaluator.default_as(evaluator.ort_evaluator):
        try:
            expected_outputs = node.target(*torch_args, **torch_kwargs)  # type: ignore[operator]
        except IndexError as index_error:
            # TODO(titaiwang): How to bound indices/dim: INT64
            warnings.warn(
                f"\nBypass the test of running on PyTorch Op {node.target} with "
                f"IndexError: \n{index_error}.\n This is possibly raised by "
                f"unsupported input args of randomnized dim/indices(INT64).\n"
            )
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.with_additional_message(
                f"### Op level debug is bypassed\n"
                f"{diagnostics.decorator.format_exception_in_markdown(index_error)}"
            )
            diagnostic.with_source_exception(index_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        except RuntimeError as runtime_error:
            warnings.warn(
                f"\nFail the test of running on PyTorch Op {node.target} with "
                f"RuntimeError: \n{runtime_error}.\n"
            )
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.with_additional_message(
                f"### Op level debug fails on PyTorch\n"
                f"{diagnostics.decorator.format_exception_in_markdown(runtime_error)}"
            )
            diagnostic.with_source_exception(runtime_error)
            diagnostic.level = diagnostics.levels.WARNING
            return

        # TODO(titaiwang): Need Opschema from ONNX function to better split args/kwargs
        # Currently, we only support torch.Tensor to numpy array. Potentially, we
        # could fail on INT64. However, we don't support dims/indices INT64 validation.
        input_onnx = [
            onnx_proto_utils._convert_tensor_to_numpy(x)
            if isinstance(x, (torch.Tensor, torch.dtype, list, tuple))
            else x
            for x in torch_args
        ]
        kwargs_onnx = fx_onnx_interpreter.filter_incompatible_and_dtype_convert_kwargs(
            torch_kwargs
        )
        try:
            ort_outputs = symbolic_fn(*input_onnx, **kwargs_onnx)
        except ValueError as value_error:
            # FIXME(titaiwang): This is caused by wrongly split args/kwargs.
            # When Opschema is ready, we should follow Opschema to split args/kwargs.
            warnings.warn(
                f"\nBypass the test of running on ONNX Op {function_name} with "
                f"ValueError: \n{value_error}.\n This is possibly raised by "
                f"unsupported input args due to lack of Opschema.\n"
            )
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.with_additional_message(
                f"### Op level debug is bypassed\n"
                f"{diagnostics.decorator.format_exception_in_markdown(value_error)}"
            )
            diagnostic.with_source_exception(value_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        except RuntimeError as runtime_error:
            warnings.warn(
                f"\nFail the test of running on ONNX Op {function_name} with "
                f"RuntimeError: \n{runtime_error}.\n"
            )
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.with_additional_message(
                f"### Op level debug fails on ONNXRUNTIME:\n"
                f"{diagnostics.decorator.format_exception_in_markdown(runtime_error)}"
            )
            diagnostic.with_source_exception(runtime_error)
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
                warnings.warn(
                    f"\nSuppressed AssertionError:\n{e}.\n"
                    f"Op {node.target} has mismatch outputs. "
                    f"Please check the implementation of {function_name}.\n"
                )
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.with_additional_message(
                    f"### Validation failed\n"
                    f"{diagnostics.decorator.format_exception_in_markdown(e)}"
                )
                diagnostic.with_source_exception(e)
                diagnostic.level = diagnostics.levels.WARNING


@_beartype.beartype
def generate_random_tensors(shape: torch.Size, dtype: torch.dtype):
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
    return torch.randn(shape, dtype=dtype)


@_beartype.beartype
def _fx_args_to_torch_args(
    complete_args: List[fx_type_utils.Argument],
) -> List[fx_type_utils.Argument]:
    """Recursively convert fx args to torch args"""
    wrapped_args: List[fx_type_utils.Argument] = []
    for arg in complete_args:
        if isinstance(arg, torch.fx.Node):
            # NOTE(titaiwang): The arg type here should align to the type handled in
            # shape.inference.FakeTensorPropGetStaticShapes. Currently, we are aware
            # of FakeTensor/Tensor/SymInt/SymFloat/Symbool/int/float/bool could be in
            # arg.meta["static_shape"].
            fake_tensor = arg.meta.get("static_shape", None)
            if isinstance(fake_tensor, torch.Tensor):
                real_tensor = generate_random_tensors(
                    fake_tensor.shape, fake_tensor.dtype
                )
                wrapped_args.append(real_tensor)
            elif isinstance(fake_tensor, (int, float, bool)):
                wrapped_args.append(fake_tensor)
            elif isinstance(fake_tensor, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                raise ValueError(
                    f"Unexpected input argument Sym type found inside fx.Node. arg: {arg}; "
                    f"arg.meta['static_shape']: {fake_tensor}; type(arg.meta['static_shape']): "
                    f"{type(fake_tensor)}. Sym type is not supported in op_level_debug."
                )
            else:
                raise ValueError(
                    f"Unexpected input argument type found inside fx.Node. arg: {arg}; "
                    f"arg.meta['static_shape']: {fake_tensor}; type(arg.meta['static_shape']): "
                    f"{type(fake_tensor)}."
                )
        elif isinstance(arg, Sequence):
            wrapped_args.append(_fx_args_to_torch_args(arg))
        elif isinstance(arg, (int, float, torch.dtype)) or arg is None:
            wrapped_args.append(arg)
        else:
            raise ValueError(
                f"Unexpected input argument type is found in node arguments. arg: {arg}; "
            )

    return wrapped_args


@_beartype.beartype
def wrap_fx_args_as_torch_args(
    complete_args: List[fx_type_utils.Argument],
    complete_kwargs: Dict[str, fx_type_utils.Argument],
) -> Tuple[tuple, dict]:
    """Prepare torch format args and kwargs for op-level validation by using fake tensor to create real tensor to feed in ops"""

    # NOTE: This function only supports FakeTensor with concrete shapes
    torch_args: List[fx_type_utils.Argument] = _fx_args_to_torch_args(complete_args)
    torch_kwargs = complete_kwargs
    return tuple(torch_args), torch_kwargs
