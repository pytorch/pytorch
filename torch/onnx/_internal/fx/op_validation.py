from __future__ import annotations

import warnings

from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]

import torch
import torch.fx

from torch.onnx import _constants, _type_utils
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.onnx._internal.fx import diagnostics
from torch.onnx._internal.fx.passes import fx_to_onnxscript
from torch.utils import _pytree


@_beartype.beartype
def validate_op_between_ort_torch(
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, Callable],
    torch_args: tuple,
    torch_kwargs: dict,
):
    """Validate the op between ONNX Runtime and PyTorch."""
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)
    # trace_only function is regular python function
    function_name = (
        symbolic_fn.name
        if isinstance(symbolic_fn, onnxscript.OnnxFunction)
        else symbolic_fn.__name__
    )
    try:
        with evaluator.default_as(evaluator.ort_evaluator):
            expected_outputs = node.target(*torch_args, **torch_kwargs)  # type: ignore[operator]
            # TODO(titaiwang): Expose _convert_tensor_to_numpy and _convert_kwargs_for_onnx?
            input_onnx = [
                onnx_proto_utils._convert_tensor_to_numpy(x) for x in torch_args
            ]
            kwargs_onnx = fx_to_onnxscript.filter_incompatible_and_dtype_convert_kwargs(
                torch_kwargs
            )
            ort_outputs = symbolic_fn(*input_onnx, **kwargs_onnx)

            # TODO: add pytree structure comparison.
            flattened_torch_outputs, _ = _pytree.tree_flatten(expected_outputs)
            flattened_function_outputs, _ = _pytree.tree_flatten(ort_outputs)

            assert flattened_torch_outputs
            assert len(flattened_torch_outputs) == len(flattened_function_outputs)

            for torch_output, function_output in zip(
                flattened_torch_outputs, flattened_function_outputs
            ):
                try:
                    if not isinstance(function_output, np.ndarray):
                        # An onnxscript tensor
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
                    diagnostic = diagnostics.export_context().inflight_diagnostic()
                    diagnostic.with_additional_message(
                        f"### Validation failed\n"
                        f"{diagnostics.decorator.format_exception_in_markdown(e)}"
                    )
                    diagnostic.level = diagnostics.levels.ERROR
    except Exception as e:
        warnings.warn(
            f"\nORT fails to run on Op {node.target} with error: \n{e}.\n"
            f"Please check the implementation of {function_name}.\n"
        )
        diagnostic = diagnostics.export_context().inflight_diagnostic()
        diagnostic.with_additional_message(
            f"### Validation failed\n"
            f"{diagnostics.decorator.format_exception_in_markdown(e)}"
        )
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
def wrap_fx_args_as_torch_args(
    complete_args: List[_type_utils.Argument],
    complete_kwargs: Dict[str, _type_utils.Argument],
) -> Tuple[tuple, dict]:
    """Prepare torch format args and kwargs for op-level validation by using fake tensor to create real tensor to feed in ops"""

    # NOTE: This function only supports FakeTensor with concrete shapes
    torch_args: List[_type_utils.Argument] = []
    for arg in complete_args:
        if isinstance(arg, torch.fx.Node):
            fake_tensor = arg.meta["val"]
            if isinstance(fake_tensor, Sequence):
                for meta_value in fake_tensor:
                    real_tensor = generate_random_tensors(
                        meta_value.shape, meta_value.dtype
                    )
                    torch_args.append(real_tensor)
            elif isinstance(fake_tensor, torch.Tensor):
                real_tensor = generate_random_tensors(
                    fake_tensor.shape, fake_tensor.dtype
                )
                torch_args.append(real_tensor)
        else:
            torch_args.append(arg)
    torch_kwargs = complete_kwargs
    return (tuple(torch_args), torch_kwargs)
