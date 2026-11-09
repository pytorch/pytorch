# Owner(s): ["module: onnx"]
"""Test op correctness by comparing with PyTorch results.

## Usage

1. Set the env var CATCH_ORT_SEGFAULT to catch segfaults from ONNX Runtime.

## How to add a new operator test

This test use PyTorch's OpInfo mechanism to generate test cases for each operator.
You may find all OpInfos in https://github.com/pytorch/pytorch/blob/7ec0d6f006fdd2c9b978dc6aa4923144684a3f51/torch/testing/_internal/common_methods_invocations.py#L8804

1. To enable test cases for an operator
    Add a `TorchLibOpInfo` entry to `TORCH_LIB_OPINFO` in `ops_test_data.py`.
    Specify `complex` if the function is designed for complex inputs.

    The `op_info_name` in `TorchLibOpInfo` needs to be unique in the TORCH_LIB_OPINFO
    list, but complex=True ops can share the same name with non-complex ops
    because they are tested separately.

2. Add `.skip` and/or `.xfail` to skip or xfail tests.
    Prefer xfail over skip when possible because that allows us to monitor the behavior
    and update the test will it passes.

    2a. If a test is now failing because of xpass, because some previous errors
    are now fixed, removed the corresponding xfail.

3. If sample inputs of the OpInfo needs to be adjusted to fit the aten signature, create an input
wrangler function. See `_mean_input_wrangler` for an example.

4. To test different ONNX functions that are registered as overloads of the same
    op, use `ops_test_common.duplicate_opinfo` to create new OpInfo with new names and map each
    to one overload.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Any, Self, TYPE_CHECKING

import numpy as np
import ops_test_common

import torch
from torch.onnx._internal.exporter._torchlib.ops import core as core_ops, nn as nn_ops
from torch.testing._internal import common_dtype, common_methods_invocations
from torch.testing._internal.opinfo import (
    core as opinfo_core,
    definitions as opinfo_definitions,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Collection


# Create a copy of the op_db to modify
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

# Append extra op_db into the op database for testing
OPS_DB.extend(opinfo_definitions.signal.op_db)


def _sample_inputs_local_scalar_dense(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    for shape in ((), (1,), (3,), (1, 1), (1, 2), (2, 1), (1, 1, 1), (2, 2, 2)):
        tensor = torch.testing.make_tensor(
            shape,
            low=0,
            high=1,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )
        yield opinfo_core.SampleInput(tensor)


OPS_DB.extend(
    (
        opinfo_core.BinaryUfuncInfo(
            "ops.aten.floor_divide",
            aten_name="floor_divide",
            dtypes=common_dtype.all_types_and_half(),
            rhs_make_tensor_kwargs=dict(exclude_zero=True),
        ),
        opinfo_core.OpInfo(
            "ops.aten._local_scalar_dense",
            aten_name="_local_scalar_dense",
            dtypes=common_dtype.all_types_and(torch.bool),
            sample_inputs_func=_sample_inputs_local_scalar_dense,
            supports_out=False,
        ),
    )
)


@dataclasses.dataclass
class TorchLibOpInfo:
    """A dataclass to store the information to test an torchlib op."""

    # The name of the op_info, e.g. "add"
    op_info_name: str
    # The torchlib ONNX Function to test
    op: Callable[..., Any]
    # The input wrangler function to adjust the input to fit the aten signature
    input_wrangler: (
        Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]] | None
    ) = None
    # Whether the op is non-deterministic
    nondeterministic: bool = False
    # Whether to compare the shape only for the output[index]
    # For example: (1,2) means compare value for output[0] and shape for output[1] and [2]
    # We may be able to combine this with the nondeterministic option
    compare_shape_only_for_output: tuple[int, ...] = ()
    # Whether the function is designed for complex inputs
    complex: bool = False
    # Eligibility for numerical tests only; native dispatch does not inspect dtypes.
    supported_dtypes: tuple[torch.dtype, ...] | None = None
    # The ONNX opset version in which the function was introduced.
    # Its specifies the minimum ONNX opset version required to use the function.
    # It ensures that the function is only used when the target ONNX opset version
    # is compatible. For example, if `opset_introduced=20`, the function will only
    # be used when exporting to ONNX models targeting opset version 20 or higher.
    opset_introduced: int = 18
    # The acceptable tolerance of the inference result difference between PyTorch and ORT.
    # Format: {dtype: (rtol, atol)}.
    # For example: {torch.float16: (1e-3, 1e-3)}
    tolerance: dict[torch.dtype, tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )
    # Expected skips or fails for the test and/or subtests
    skips_or_fails: list[ops_test_common.DecorateMeta] = dataclasses.field(
        default_factory=list
    )

    def get_tolerance(self, dtype: torch.dtype) -> tuple[float | None, float | None]:
        """Returns the (rtol, atol) tolerance for the given dtype."""
        if (tolerance := self.tolerance.get(dtype)) is not None:
            return tolerance

        # Use the PyTorch default if not specified
        # https://pytorch.org/docs/stable/testing.html
        return (None, None)

    def skip(
        self,
        variant_name: str = "",
        *,
        reason: str,
        dtypes: Collection[torch.dtype] | None = None,
        device_type: str | None = None,
        matcher: Callable[[Any], Any] | None = None,
        enabled_if: bool = True,
        test_class_name: str | None = None,
    ) -> Self:
        """Skips an OpInfo test.

        Args:
            variant_name: Optional OpInfo variant_test_name.
            reason: The reason for skipping.
            dtypes: The dtypes to skip.
            device_type: Device type. E.g. "cpu", "cuda".
            matcher: A function that matches the test sample input. It is used only when
                the skip is in the SKIP_XFAIL_SUBTESTS list.
            enabled_if: Whether the skip is enabled.
            test_class_name: The test class name to apply the skip to. If None, the skip
                is applied to all test classes.
        """
        self.skips_or_fails.append(
            ops_test_common.skip(
                self.op_info_name,
                variant_name,
                reason=reason,
                dtypes=dtypes,
                device_type=device_type,
                matcher=matcher,
                enabled_if=enabled_if,
                test_class_name=test_class_name,
            )
        )
        return self

    def xfail(
        self,
        variant_name: str = "",
        *,
        reason: str,
        dtypes: Collection[torch.dtype] | None = None,
        device_type: str | None = None,
        matcher: Callable[[Any], Any] | None = None,
        enabled_if: bool = True,
        test_class_name: str | None = None,
    ) -> Self:
        """Expects an OpInfo test to fail.

        Args:
            variant_name: Optional OpInfo variant_test_name.
            reason: The reason for the failure.
            dtypes: The dtypes to expect the failure
            device_type: Device type. E.g. "cpu", "cuda"..
            matcher: A function that matches the test sample input. It is used only when
                the xfail is in the SKIP_XFAIL_SUBTESTS list.
            enabled_if: Whether the xfail is enabled.
            test_class_name: The test class name to apply the xfail to. If None, the
                xfail is applied to all test classes.
        """
        self.skips_or_fails.append(
            ops_test_common.xfail(
                self.op_info_name,
                variant_name,
                reason=reason,
                dtypes=dtypes,
                device_type=device_type,
                matcher=matcher,
                enabled_if=enabled_if,
                test_class_name=test_class_name,
            )
        )
        return self


# Modify this section ##########################################################


def _amin_amax_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "dim" not in kwargs:
        # Supply an empty dim to match the aten signature
        kwargs["dim"] = np.array([], dtype=np.int64)
    else:
        # Convert dim to a numpy array
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64).reshape((-1,))
    return args, kwargs


def _avg_pool_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "dim" not in kwargs:
        if len(args) > 6:
            kwargs["divisor_override"] = args.pop(6)
        if len(args) > 5:
            kwargs["count_include_pad"] = args.pop(5)
        if len(args) > 4:
            kwargs["ceil_mode"] = args.pop(4)
        if len(args) > 3:
            padding = args.pop(3)
            if isinstance(padding, np.ndarray):
                # Cannot using list(padding) here, because the element will be numpy.int64 instead of int
                padding = padding.tolist()
            kwargs["padding"] = padding
        if len(args) > 2:
            stride = args.pop(2)
            if isinstance(stride, np.ndarray):
                stride = stride.tolist()
            kwargs["stride"] = stride
        kernel_size = args.pop(1)
        if isinstance(kernel_size, np.ndarray):
            kernel_size = kernel_size.tolist()
        kwargs["kernel_size"] = kernel_size
    return args, kwargs


def _cross_entropy_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        reduction_vals = ["none", "mean", "sum"]
        value = kwargs["reduction"]
        idx = reduction_vals.index(value)
        kwargs["reduction"] = idx
    return args, kwargs


def _dropout_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "training" in kwargs:
        kwargs["train"] = kwargs["training"]
        kwargs.pop("training")
    return args, kwargs


def _einsum_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Swap the equation and tensors to revert the special handling in the OpInfo
    return [args[1], args[0]], kwargs


def _embedding_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Remove arguments not present in the aten op signature."""
    kwargs.pop("max_norm", None)
    kwargs.pop("norm_type", None)
    return args, kwargs


def _empty_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Remove arguments not present in the aten op signature."""
    kwargs.pop("requires_grad", None)
    return args, kwargs


def _grid_sample_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Convert string attribute to int as input
    inter_mode_options = {"bilinear": 0, "nearest": 1, "bicubic": 2}
    padding_mode_options = {"zeros": 0, "border": 1, "reflection": 2}
    args.append(inter_mode_options[kwargs["mode"]])
    args.append(padding_mode_options[kwargs["padding_mode"]])
    args.append(kwargs["align_corners"])
    kwargs.clear()
    return args, kwargs


def _im2col_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Move kernel_size, dilation, padding and stride from args to kwargs
    if len(args) == 5:
        # Handle stride
        stride = args.pop()
        if isinstance(stride, np.ndarray):  # convert stride to list[int]
            stride = stride.tolist()
        kwargs["stride"] = stride
        # Handle padding
        padding = args.pop()
        if isinstance(padding, np.ndarray):  # convert padding to list[int]
            padding = padding.tolist()
        kwargs["padding"] = padding
        # Handle dilation
        dilation = args.pop()
        if isinstance(dilation, np.ndarray):  # convert dilation to list[int]
            dilation = dilation.tolist()
        kwargs["dilation"] = dilation
    # Handle kernel_size
    kernel_size = args.pop()
    if isinstance(kernel_size, np.ndarray):  # convert kernel_size to list[int]
        kernel_size = kernel_size.tolist()
    kwargs["kernel_size"] = kernel_size

    return args, kwargs


def _index_put_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args[1] = [np.array(elem) for elem in args[1]]
    return args, kwargs


def _max_pool_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Remove return_indices argument because this op doesn't accept it
    kwargs.pop("return_indices", None)
    return args, kwargs


def _mean_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the dims as tensor
    if "dim" in kwargs:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
    return args, kwargs


def _mse_loss_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        reduction_vals = ["none", "mean", "sum"]  # [0,1,2], default=1
        value = kwargs["reduction"]
        idx = reduction_vals.index(value)
        kwargs["reduction"] = idx
    return args, kwargs


def _nll_loss_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        # aten_nll_loss can only accept integer argument instead of string
        reduction_vals = ["none", "mean", "sum"]
        value = kwargs["reduction"]
        kwargs["reduction"] = reduction_vals.index(value)
    return args, kwargs


def _nonzero_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    kwargs.pop("as_tuple", None)
    return args, kwargs


def _reflection_pad2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'reflect' arg
    return args, kwargs


def _replication_pad2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'replicate' arg
    return args, kwargs


def _replication_pad3d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'replicate' arg
    return args, kwargs


def _roll_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if len(args) >= 3:
        if isinstance(args[2], np.ndarray):  # convert dims to list[int]
            # Change dims from args to kwargs to keep tuple/list type
            dims = args.pop(2)
            kwargs["dims"] = dims.tolist()
        elif isinstance(args[2], int):  # convert dims to list[int]
            dims = args.pop(2)
            kwargs["dims"] = []
            kwargs["dims"].append(dims)
    if len(args) >= 2:
        if isinstance(args[1], np.ndarray):  # convert shift to list[int]
            shifts = args.pop(1)
            kwargs["shifts"] = shifts.tolist()
        elif isinstance(args[1], int):
            shifts = args.pop(1)
            kwargs["shifts"] = []
            kwargs["shifts"].append(shifts)
    return args, kwargs


def _scalar_tensor_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    kwargs.pop("requires_grad", None)
    return args, kwargs


def _scatter_reduce_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Put the string into kwargs, otherwise FullGraph mode could not find get 'reduce' argument
    kwargs["reduce"] = args.pop(4)
    return args, kwargs


def _sum_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if kwargs.get("dim") is not None:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
    return args, kwargs


def _unflatten_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args[1] = np.array(args[1], dtype=np.int64)
    return args, kwargs


def _where_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # The aten::where op takes condition, x, y as inputs
    # Swap the first two inputs
    args[0], args[1] = args[1], args[0]
    return args, kwargs


# Ops to be tested for numerical consistency between onnx and pytorch
# Find the names of the OpInfos in torch/testing/_internal/common_methods_invocations.py
TESTED_TORCHLIB_OPS: tuple[TorchLibOpInfo, ...] = (
    TorchLibOpInfo("abs", core_ops.aten_abs),
    TorchLibOpInfo("abs", core_ops.aten_abs_complex, complex=True),
    TorchLibOpInfo("gelu_op20", nn_ops.aten_gelu_opset20, opset_introduced=20),
    TorchLibOpInfo(
        "nn.functional.group_norm", nn_ops.aten_group_norm, opset_introduced=21
    ).skip(
        reason="ONNX Runtime does not support zero sized inputs for GroupNorm",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo(
        "nn.functional.rms_norm", nn_ops.aten_rms_norm, opset_introduced=23
    ).skip(
        reason="ONNX Runtime does not support <1d inputs or zero sized inputs for RMSNorm",
        matcher=lambda sample: len(sample.input.shape) < 2 or sample.input.numel() == 0,
    ),
)


# Source-supported dtypes are test metadata, not exporter dispatch constraints.
_FLOAT_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
)
_FLOAT_HIGH_PRECISION_DTYPES = (
    torch.float32,
    torch.float64,
)
_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_REAL_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_REAL_UINT8_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
_TENSOR_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_COMPLEX_DTYPES = (
    torch.complex32,
    torch.complex64,
    torch.complex128,
)
TESTED_TORCHLIB_OPS += (
    TorchLibOpInfo(
        "ops.aten._local_scalar_dense",
        core_ops.aten__local_scalar_dense,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo("acos", core_ops.aten_acos, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("acosh", core_ops.aten_acosh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "add",
        core_ops.aten_add,
        tolerance={torch.float16: (1e-3, 1e-3)},
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "add", core_ops.aten_add_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo(
        "addcdiv",
        core_ops.aten_addcdiv,
        tolerance={torch.float16: (3e-2, 1e-3)},
        supported_dtypes=_FLOAT_DTYPES,
    ),
    TorchLibOpInfo(
        "addcmul",
        core_ops.aten_addcmul,
        tolerance={torch.float16: (4e-3, 3e-3)},
        supported_dtypes=_REAL_DTYPES,
    ),
    TorchLibOpInfo("angle", core_ops.aten_angle, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "angle",
        core_ops.aten_angle_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("asin", core_ops.aten_asin, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("asinh", core_ops.aten_asinh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("atan", core_ops.aten_atan, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("atan2", core_ops.aten_atan2, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("atanh", core_ops.aten_atanh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "bitwise_and", core_ops.aten_bitwise_and, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "bitwise_left_shift",
        core_ops.aten_bitwise_left_shift,
        supported_dtypes=_INT_DTYPES,
    ),
    TorchLibOpInfo(
        "bitwise_not", core_ops.aten_bitwise_not, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "bitwise_or", core_ops.aten_bitwise_or, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "bitwise_right_shift",
        core_ops.aten_bitwise_right_shift,
        supported_dtypes=_INT_DTYPES,
    ),
    TorchLibOpInfo(
        "bitwise_xor", core_ops.aten_bitwise_xor, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo("ceil", core_ops.aten_ceil, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "clamp_max", core_ops.aten_clamp_max_tensor, supported_dtypes=_REAL_DTYPES
    ).skip(
        reason="Size 0 inputs are not handled by design",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo(
        "clamp_min", core_ops.aten_clamp_min_tensor, supported_dtypes=_REAL_DTYPES
    ).skip(
        reason="Size 0 inputs are not handled by design",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo("clone", core_ops.aten_clone, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "clone",
        core_ops.aten_clone_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("conj", core_ops.aten_conj, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "conj",
        core_ops.aten_conj_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("cos", core_ops.aten_cos, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("cosh", core_ops.aten_cosh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("deg2rad", core_ops.aten_deg2rad, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("div", core_ops.aten_div, supported_dtypes=_FLOAT_DTYPES).skip(
        matcher=lambda sample: sample.kwargs.get("rounding_mode") is not None,
        reason="this variation does not take the rounding_mode argument",
    ),
    TorchLibOpInfo("true_divide", core_ops.aten_div, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "true_divide",
        core_ops.aten_div_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo(
        "div_mode", core_ops.aten_div_mode, supported_dtypes=_REAL_DTYPES
    ).skip(
        variant_name="no_rounding_mode",
        reason="this variation requires the rounding_mode argument",
    ),
    TorchLibOpInfo("eq", core_ops.aten_eq, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("exp", core_ops.aten_exp, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("exp2", core_ops.aten_exp2, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("floor", core_ops.aten_floor, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "ops.aten.floor_divide",
        core_ops.aten_floor_divide,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo("fmod", core_ops.aten_fmod, supported_dtypes=_REAL_UINT8_DTYPES),
    TorchLibOpInfo("frac", core_ops.aten_frac, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("ge", core_ops.aten_ge, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("gt", core_ops.aten_gt, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("isclose", core_ops.aten_isclose, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo(
        "isfinite",
        core_ops.aten_isfinite,
        supported_dtypes=_FLOAT_HIGH_PRECISION_DTYPES,
    ),
    TorchLibOpInfo("isinf", core_ops.aten_isinf, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("isnan", core_ops.aten_isnan, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("isneginf", core_ops.aten_isneginf, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("isposinf", core_ops.aten_isposinf, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("log", core_ops.aten_log, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("le", core_ops.aten_le, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "lerp",
        core_ops.aten_lerp,
        tolerance={torch.float16: (2e-3, 2e-1)},
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo("log10", core_ops.aten_log10, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("log1p", core_ops.aten_log1p, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("log2", core_ops.aten_log2, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "logaddexp",
        core_ops.aten_logaddexp,
        tolerance={torch.float16: (1, 1e-4)},
        supported_dtypes=_FLOAT_DTYPES,
    ),
    TorchLibOpInfo(
        "logaddexp2",
        core_ops.aten_logaddexp2,
        tolerance={torch.float16: (2e-2, 6e-4)},
        supported_dtypes=_FLOAT_DTYPES,
    ),
    TorchLibOpInfo("lt", core_ops.aten_lt, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("maximum", core_ops.aten_maximum, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("minimum", core_ops.aten_minimum, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("mul", core_ops.aten_mul, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "mul", core_ops.aten_mul_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo("ne", core_ops.aten_ne, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo("neg", core_ops.aten_neg, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo(
        "neg", core_ops.aten_neg_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo("pow", core_ops.aten_pow, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo("rad2deg", core_ops.aten_rad2deg, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "reciprocal", core_ops.aten_reciprocal, supported_dtypes=_FLOAT_DTYPES
    ),
    TorchLibOpInfo(
        "reciprocal",
        core_ops.aten_reciprocal_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo(
        "remainder", core_ops.aten_remainder, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo("round", core_ops.aten_round, supported_dtypes=_FLOAT_DTYPES).skip(
        matcher=lambda sample: sample.kwargs.get("decimals") is not None,
        reason="this Aten overload only support one tensor as input and one int as args by design",
    ),
    TorchLibOpInfo(
        "round_decimals", core_ops.aten_round_decimals, supported_dtypes=_FLOAT_DTYPES
    ),
    TorchLibOpInfo("rsqrt", core_ops.aten_rsqrt, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("sign", core_ops.aten_sign, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("sin", core_ops.aten_sin, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("sinh", core_ops.aten_sinh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("sqrt", core_ops.aten_sqrt, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo(
        "sub",
        core_ops.aten_sub,
        tolerance={torch.float16: (2e-3, 1e-3)},
        supported_dtypes=_REAL_DTYPES,
    ),
    TorchLibOpInfo(
        "sub", core_ops.aten_sub_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo("tan", core_ops.aten_tan, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("tanh", core_ops.aten_tanh, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("trunc", core_ops.aten_trunc, supported_dtypes=_FLOAT_DTYPES),
    TorchLibOpInfo("clamp", core_ops.aten_clamp_tensor, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo(
        "logical_and", core_ops.aten_logical_and, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "logical_not", core_ops.aten_logical_not, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "logical_or", core_ops.aten_logical_or, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "logical_xor", core_ops.aten_logical_xor, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "logit",
        core_ops.aten_logit,
        tolerance={torch.float16: (1e-1, 7e-4)},
        supported_dtypes=_FLOAT_DTYPES,
    ),
)

ops_test_common.duplicate_opinfo(OPS_DB, "all", ("all_dim", "all_dims"))
ops_test_common.duplicate_opinfo(OPS_DB, "any", ("any_dim", "any_dims"))
ops_test_common.duplicate_opinfo(
    OPS_DB, "arange", ("arange_start", "arange_start_step")
)
ops_test_common.duplicate_opinfo(OPS_DB, "atleast_1d", ("atleast_1d_Sequence",))
ops_test_common.duplicate_opinfo(OPS_DB, "atleast_2d", ("atleast_2d_Sequence",))
ops_test_common.duplicate_opinfo(OPS_DB, "atleast_3d", ("atleast_3d_Sequence",))
ops_test_common.duplicate_opinfo(
    OPS_DB,
    "bitwise_left_shift",
    (
        "bitwise_left_shift_int8",
        "bitwise_left_shift_int16",
        "bitwise_left_shift_int32",
        "bitwise_left_shift_int64",
    ),
)
ops_test_common.duplicate_opinfo(
    OPS_DB,
    "bitwise_right_shift",
    (
        "bitwise_right_shift_int8",
        "bitwise_right_shift_int16",
        "bitwise_right_shift_int32",
        "bitwise_right_shift_int64",
    ),
)
ops_test_common.duplicate_opinfo(OPS_DB, "cat", ("concat", "concatenate"))
ops_test_common.duplicate_opinfo(OPS_DB, "clone", ("lift_fresh_copy",))
ops_test_common.duplicate_opinfo(OPS_DB, "diagonal", ("diagonal_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "div", ("div_mode", "div_mode_int"))
ops_test_common.duplicate_opinfo(OPS_DB, "ge", ("ge_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "gt", ("gt_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "index_put", ("index_put_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "le", ("le_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "lt", ("lt_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "max", ("max_dim",))
ops_test_common.duplicate_opinfo(OPS_DB, "maximum", ("maximum_bool",))
ops_test_common.duplicate_opinfo(OPS_DB, "mean", ("mean_dim",))
ops_test_common.duplicate_opinfo(OPS_DB, "min", ("min_dim",))
ops_test_common.duplicate_opinfo(OPS_DB, "minimum", ("minimum_bool",))
ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.pad",
    (
        "nn.functional.reflection_pad2d",
        "nn.functional.replication_pad2d",
        "nn.functional.replication_pad3d",
    ),
)
ops_test_common.duplicate_opinfo(OPS_DB, "nn.functional.gelu", ("gelu_op20",))
ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.scaled_dot_product_attention",
    ("nn.functional.scaled_dot_product_attention_bool_mask",),
)
ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.celu",
    ("nn.functional.celu_type_promoted",),
)
ops_test_common.duplicate_opinfo(
    OPS_DB, "ops.aten._log_softmax", ("ops.aten._log_softmax_half",)
)
ops_test_common.duplicate_opinfo(
    OPS_DB, "ops.aten._softmax", ("ops.aten._softmax_half",)
)
ops_test_common.duplicate_opinfo(OPS_DB, "prod", ("prod_dim_int",))
ops_test_common.duplicate_opinfo(OPS_DB, "round", ("round_decimals",))
ops_test_common.duplicate_opinfo(OPS_DB, "squeeze", ("squeeze_dim",))
ops_test_common.duplicate_opinfo(OPS_DB, "view_as_complex", ("view_as_complex_copy",))
ops_test_common.duplicate_opinfo(OPS_DB, "view_as_real", ("view_as_real_copy",))

# MARK: End edits here


# These ops are not deterministic, so we check shape and dtype only
NONDETERMINISTIC_OPS: frozenset[str] = frozenset(
    info.op_info_name for info in TESTED_TORCHLIB_OPS if info.nondeterministic
)

COMPARE_SHAPE_ONLY_OPS: dict[
    str,
    set,
] = {
    info.op_info_name: set(info.compare_shape_only_for_output)
    for info in TESTED_TORCHLIB_OPS
}

TORCHLIB_OPINFO_MAPPING: dict[
    str,
    TorchLibOpInfo,
] = {info.op_info_name: info for info in TESTED_TORCHLIB_OPS if not info.complex}

TESTED_OPS = frozenset(TORCHLIB_OPINFO_MAPPING)

EXPECTED_SKIPS_OR_FAILS: tuple[ops_test_common.DecorateMeta, ...] = tuple(
    functools.reduce(
        # Flatten the list
        lambda a, b: [*a, *b],
        [
            [meta for meta in info.skips_or_fails if meta.matcher is None]
            for info in TESTED_TORCHLIB_OPS
        ],
    )
)

SKIP_XFAIL_SUBTESTS: tuple[ops_test_common.DecorateMeta, ...] = tuple(
    functools.reduce(
        # Flatten the list
        lambda a, b: [*a, *b],
        [
            [meta for meta in info.skips_or_fails if meta.matcher is not None]
            for info in TESTED_TORCHLIB_OPS
        ],
    )
)

# MARK: Complex supported functions
COMPLEX_FUNCTION_MAPPING: dict[
    str,
    TorchLibOpInfo,
] = {info.op_info_name: info for info in TESTED_TORCHLIB_OPS if info.complex}


# Call dir(torch.ops.prims) and compare with entries in OPS_DB to create OpInfo for newly added prims ops
PRIMS_OPS_WITH_OP_INFO = (
    "abs",
    "acos",
    "acosh",
    "add",
    "amax",
    "amin",
    "as_strided",
    "as_strided_scatter",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cat",
    "ceil",
    "clone",
    "conj",
    "conj_physical",
    "cos",
    "cosh",
    "digamma",
    "div",
    "empty",
    "eq",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fill",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "full",
    "full_like",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "isfinite",
    "le",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "normal",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "reshape",
    "round",
    "rsqrt",
    "scalar_tensor",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "squeeze",
    "sub",
    "sum",
    "svd",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "uniform",
    "where",
)

for op in PRIMS_OPS_WITH_OP_INFO:
    # Duplicate opinfo for prim ops. The new names all start with "prims_". E.g. "abs" -> "prims_abs".
    ops_test_common.duplicate_opinfo_for_prims(OPS_DB, op)

# Duplicate cases where the prims op name is different from the torch op name
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "i0", "bessel_i0")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.bessel_j0", "bessel_j0")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.bessel_j1", "bessel_j1")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.erfcx", "erfcx")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i0e", "bessel_i0e")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i1", "bessel_i1")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i1e", "bessel_i1e")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.ndtri", "ndtri")
ops_test_common.duplicate_opinfo_for_prims(
    OPS_DB, "special.spherical_bessel_j0", "spherical_bessel_j0"
)
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.zeta", "zeta")

OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(meta.op_name for meta in SKIP_XFAIL_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)
# Assert all ops in OPINFO_FUNCTION_MAPPING are in the OPS_DB
if not TESTED_OPS.issubset(ALL_OPS_IN_DB):
    raise AssertionError(f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB")
if not NONDETERMINISTIC_OPS.issubset(TESTED_OPS):
    raise AssertionError(f"{NONDETERMINISTIC_OPS - TESTED_OPS} not in TESTED_OPS")
