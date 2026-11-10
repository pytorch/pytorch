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
import sys
from typing import Any, Self, TYPE_CHECKING

import extra_opinfo
import numpy as np
import ops_test_common

import torch
from torch.onnx._internal.exporter._torchlib.ops import (
    core as core_ops,
    embedding as embedding_ops,
    factories as factories_ops,
    fft as fft_ops,
    linalg as linalg_ops,
    neural as neural_ops,
    nn as nn_ops,
    normalization as normalization_ops,
    prims as prims_ops,
    residual as residual_ops,
    sequences as sequences_ops,
    special as special_ops,
    vision as vision_ops,
)
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
OPS_DB.extend(extra_opinfo.OPS_DB)


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


# Source-supported dtypes are test metadata, not exporter dispatch constraints.
TESTED_TORCHLIB_OPS += (
    TorchLibOpInfo(
        "all_dim", core_ops.aten_all_dim, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: (
            not (len(sample.kwargs) > 0) or isinstance(sample.kwargs.get("dim"), tuple)
        ),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design. dim must be an integer",
    ),
    TorchLibOpInfo(
        "all_dims", core_ops.aten_all_dims, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: not isinstance(sample.kwargs.get("dim"), tuple),
        reason="this overload requires dim to be a tuple",
    ),
    TorchLibOpInfo("all", core_ops.aten_all, supported_dtypes=_TENSOR_DTYPES).skip(
        matcher=lambda sample: len(sample.kwargs) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo(
        "amax",
        core_ops.aten_amax,
        input_wrangler=_amin_amax_input_wrangler,
        supported_dtypes=_REAL_UINT8_DTYPES,
    ),
    TorchLibOpInfo(
        "amin",
        core_ops.aten_amin,
        input_wrangler=_amin_amax_input_wrangler,
        supported_dtypes=_REAL_UINT8_DTYPES,
    ),
    TorchLibOpInfo("any", core_ops.aten_any, supported_dtypes=_TENSOR_DTYPES).skip(
        matcher=lambda sample: len(sample.kwargs) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo(
        "any_dim", core_ops.aten_any_dim, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: (
            not (len(sample.kwargs) > 0) or isinstance(sample.kwargs.get("dim"), tuple)
        ),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design. dim must be an integer",
    ),
    TorchLibOpInfo(
        "any_dims", core_ops.aten_any_dims, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: not isinstance(sample.kwargs.get("dim"), tuple),
        reason="this overload requires dim to be a tuple",
    ),
    TorchLibOpInfo(
        "atleast_1d", core_ops.aten_atleast_1d, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "atleast_2d", core_ops.aten_atleast_2d, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "atleast_3d", core_ops.aten_atleast_3d, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "broadcast_to", core_ops.aten_broadcast_to, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo("diagonal", core_ops.aten_diagonal, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("expand", core_ops.aten_expand, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "expand_as", core_ops.aten_expand_as, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo("flip", core_ops.aten_flip, supported_dtypes=_TENSOR_DTYPES).skip(
        reason="fixme: size 0 inputs are not handled yet",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo(
        "flip",
        core_ops.aten_flip_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("flatten", core_ops.aten_flatten, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("gather", core_ops.aten_gather, supported_dtypes=_REAL_DTYPES).skip(
        matcher=lambda sample: sample.input.numel() == 0 or sample.args[1].numel() == 0,
        reason="fixme: ORT does not support empty tensors as input",
    ),
    TorchLibOpInfo(
        "ops.aten.index.Tensor", core_ops.aten_index, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "ops.aten.index.Tensor.bool",
        core_ops.aten_index,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "index_put",
        core_ops.aten_index_put,
        input_wrangler=_index_put_input_wrangler,
        supported_dtypes=_REAL_DTYPES,
    ).skip(
        dtypes=(torch.float16,),
        matcher=lambda sample: sample.kwargs.get("accumulate") is True,
        reason="fixme: ORT only supports float32 when accumulate is True:  MLFloat16 data type is not supported with ScatterND when reduction is 'add'",
    ),
    TorchLibOpInfo(
        "ops.aten.index_put", core_ops.aten_index_put, supported_dtypes=_REAL_DTYPES
    ),
    TorchLibOpInfo(
        "ops.aten._unsafe_index_put",
        core_ops.aten_index_put,
        supported_dtypes=_REAL_DTYPES,
    ),
    TorchLibOpInfo(
        "index_select", core_ops.aten_index_select, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "logsumexp",
        core_ops.aten_logsumexp,
        tolerance={torch.float16: (2e-2, 1e-4)},
        supported_dtypes=_FLOAT_DTYPES,
    ),
    TorchLibOpInfo(
        "masked_fill", core_ops.aten_masked_fill, supported_dtypes=_TENSOR_DTYPES
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not have an implementation for Where with bool inputs.",
    ),
    TorchLibOpInfo(
        "masked_scatter", core_ops.aten_masked_scatter, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "ops.aten.masked_scatter",
        core_ops.aten_masked_scatter,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "mean",
        core_ops.aten_mean,
        input_wrangler=_mean_input_wrangler,
        supported_dtypes=_REAL_DTYPES,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is not None,
        reason="this Aten overload only accept 1 inputs: self",
    ),
    TorchLibOpInfo(
        "ops.aten.mean.dtype", core_ops.aten_mean, supported_dtypes=_REAL_DTYPES
    ),
    TorchLibOpInfo(
        "mean",
        core_ops.aten_mean_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is not None,
        reason="this Aten overload only accepts self and dtype",
    ),
    TorchLibOpInfo(
        "mean_dim",
        core_ops.aten_mean_dim_complex,
        input_wrangler=_mean_input_wrangler,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is None,
        reason="the dimension-free overload is tested separately",
    ),
    TorchLibOpInfo(
        "mean_dim",
        core_ops.aten_mean_dim,
        input_wrangler=_mean_input_wrangler,
        supported_dtypes=_REAL_DTYPES,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is None,
        reason="this Aten overload can accept 2 inputs:(self, dim)",
    ),
    TorchLibOpInfo("mH", core_ops.aten_mH, supported_dtypes=_REAL_UINT8_DTYPES),
    TorchLibOpInfo(
        "mH", core_ops.aten_mH_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo(
        "min_dim", core_ops.aten_min_dim, supported_dtypes=_REAL_DTYPES
    ).xfail(
        matcher=lambda sample: (
            len(sample.args) == 0
            or (len(sample.args) > 0 and not isinstance(sample.args[0], int))
        ),
        reason="this ATen overload only support one tensor as input and another int as args",
    ),
    TorchLibOpInfo("min", core_ops.aten_min, supported_dtypes=_REAL_DTYPES).skip(
        matcher=lambda sample: len(sample.args) > 0,
        reason="this ATen overload only supports one tensor as input by design",
    ),
    TorchLibOpInfo("mT", core_ops.aten_mT, supported_dtypes=_REAL_UINT8_DTYPES),
    TorchLibOpInfo(
        "mT", core_ops.aten_mT_complex, complex=True, supported_dtypes=_COMPLEX_DTYPES
    ),
    TorchLibOpInfo("narrow", core_ops.aten_narrow, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "nonzero",
        core_ops.aten_nonzero,
        input_wrangler=_nonzero_input_wrangler,
        supported_dtypes=_TENSOR_DTYPES,
    )
    .xfail(
        matcher=lambda sample: sample.kwargs.get("as_tuple"),
        reason="as_tuple=True is not supported",
    )
    .xfail(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0]).",
    ),
    TorchLibOpInfo("permute", core_ops.aten_permute, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "permute",
        core_ops.aten_permute_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("prod", core_ops.aten_prod, supported_dtypes=_REAL_DTYPES).skip(
        matcher=lambda sample: (
            sample.kwargs.get("dim") is not None
            or sample.kwargs.get("keepdim") is not None
            or len(sample.args) > 0
        ),
        reason="this Aten overload only accept 1 inputs: self",
    ),
    TorchLibOpInfo(
        "prod_dim_int", core_ops.aten_prod_dim_int, supported_dtypes=_REAL_DTYPES
    ).skip(
        matcher=lambda sample: (
            (sample.kwargs.get("dim") is None and sample.kwargs.get("keepdim") is None)
            or sample.kwargs.get("dtype") != -1
        ),
        reason="this Aten overload can accept 3 inputs:(self, dim, keepdim)",
    ),
    TorchLibOpInfo("repeat", core_ops.aten_repeat, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo("reshape", core_ops.aten_reshape, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "scatter_add", core_ops.aten_scatter_add, supported_dtypes=_REAL_DTYPES
    )
    .xfail(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch. https://github.com/onnx/onnx/issues/4986",
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT error: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'",
    ),
    TorchLibOpInfo("select", core_ops.aten_select, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "select_scatter", core_ops.aten_select_scatter, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo("sort", core_ops.aten_sort, supported_dtypes=_REAL_DTYPES).xfail(
        dtypes=(torch.float16,),
        reason="fixme: Tensor-likes are not close. Tests pass for float32.",
    ),
    TorchLibOpInfo(
        "squeeze_dim", core_ops.aten_squeeze_dim, supported_dtypes=_TENSOR_DTYPES
    )
    .skip(
        matcher=lambda sample: not (
            len(sample.args) > 0 and isinstance(sample.args[0], int)
        ),
        reason="this Aten overload only support one tensor as input and one int as args by design",
    )
    .skip(
        matcher=lambda sample: (
            len(sample.input.shape) != 0 and sample.input.shape[sample.args[0]] != 1
        ),
        reason="this Aten overload only support squeeze dim with size 1",
    ),
    TorchLibOpInfo(
        "squeeze_dim",
        core_ops.aten_squeeze_dim_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    )
    .skip(
        matcher=lambda sample: not (
            len(sample.args) > 0 and isinstance(sample.args[0], int)
        ),
        reason="this Aten overload only support one tensor as input and one int as args by design",
    )
    .skip(
        matcher=lambda sample: (
            len(sample.input.shape) != 0 and sample.input.shape[sample.args[0]] != 1
        ),
        reason="this Aten overload only support squeeze dim with size 1",
    ),
    TorchLibOpInfo(
        "squeeze", core_ops.aten_squeeze, supported_dtypes=_TENSOR_DTYPES
    ).skip(
        matcher=lambda sample: len(sample.args) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo("t", core_ops.aten_t, supported_dtypes=_TENSOR_DTYPES).xfail(
        enabled_if=False,  # Native IR never creates the scripted If branches.
        reason="fixme: ORT Graph attribute inferencing failed on rank-1 input. https://github.com/onnx/onnx/issues/4986",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    TorchLibOpInfo("topk", core_ops.aten_topk, supported_dtypes=_REAL_DTYPES)
    .xfail(
        dtypes=(torch.int64, torch.int32),
        enabled_if=not ops_test_common.IS_WINDOWS,
        reason="fixme: result mismatch. https://github.com/microsoft/onnxscript/issues/853",
    )
    .skip(
        dtypes=(torch.float16,),
        reason="fixme: result mismatch. https://github.com/microsoft/onnxscript/issues/853",
    )
    .skip(
        matcher=lambda sample: len(sample.input.shape) == 0
        or sample.input.numel() == 0,
        reason="scalar inputs or empty inputs are not handled",
    ),
    TorchLibOpInfo("unflatten", core_ops.aten_unflatten, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo("unfold", core_ops.aten_unfold, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "ops.aten.unfold", core_ops.aten_unfold, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "unsqueeze", core_ops.aten_unsqueeze, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "unsqueeze",
        core_ops.aten_unsqueeze_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("view", core_ops.aten_view, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "view",
        core_ops.aten_view_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("view_as", core_ops.aten_view_as, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "view_as_complex",
        core_ops.aten_view_as_complex,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "view_as_complex_copy",
        core_ops.aten_view_as_complex_copy,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "view_as_real",
        core_ops.aten_view_as_real,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo(
        "view_as_real_copy",
        core_ops.aten_view_as_real_copy,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo(
        "view_copy", core_ops.aten_view_copy, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "where",
        core_ops.aten_where,
        input_wrangler=_where_input_wrangler,
        supported_dtypes=_TENSOR_DTYPES,
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not have an implementation for Where with bool inputs.",
    ),
    TorchLibOpInfo("argmax", core_ops.aten_argmax, supported_dtypes=_REAL_UINT8_DTYPES),
    TorchLibOpInfo("argmin", core_ops.aten_argmin, supported_dtypes=_REAL_UINT8_DTYPES),
    TorchLibOpInfo(
        "as_strided", core_ops.aten_as_strided, supported_dtypes=_TENSOR_DTYPES
    ).xfail(
        variant_name="partial_views", reason="ONNX doesn't have partial view for tensor"
    ),
    TorchLibOpInfo(
        "cumsum",
        core_ops.aten_cumsum,
        supported_dtypes=(*_FLOAT_DTYPES, torch.int32, torch.int64),
    ),
    TorchLibOpInfo("max_dim", core_ops.aten_max_dim, supported_dtypes=_REAL_DTYPES)
    .xfail(
        variant_name="reduction_with_dim",
        reason="fixme: ORT Graph attribute inferencing failed https://github.com/onnx/onnx/issues/4986",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=False,  # Native IR never creates the scripted If branches.
    )
    .xfail(
        matcher=lambda sample: (
            len(sample.args) == 0
            or (len(sample.args) > 0 and not isinstance(sample.args[0], int))
        ),
        reason="this ATen overload only support one tensor as input and another int as args",
    ),
    TorchLibOpInfo("max", core_ops.aten_max, supported_dtypes=_REAL_DTYPES).skip(
        matcher=lambda sample: len(sample.args) > 0,
        reason="this ATen overload only supports one tensor as input by design",
    ),
    TorchLibOpInfo(
        "scatter_reduce",
        core_ops.aten_scatter_reduce,
        input_wrangler=_scatter_reduce_input_wrangler,
        supported_dtypes=_REAL_DTYPES,
    )
    .xfail(
        variant_name="mean",
        dtypes=(torch.float16, torch.float64),
        reason="ORT does not implement ScatterElements additive reduction for these dtypes",
    )
    .xfail(
        variant_name="prod",
        dtypes=(torch.float16, torch.float64),
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'mul'",
    )
    .xfail(
        variant_name="sum",
        dtypes=(torch.float16, torch.float64),
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'add'",
    )
    .xfail(
        variant_name="mean",
        dtypes=(torch.bfloat16,),
        reason="onnxruntime does not support ml_dtypes.bfloat16",
    )
    .xfail(
        variant_name="prod",
        dtypes=(torch.bfloat16,),
        reason="onnxruntime does not support ml_dtypes.bfloat16",
    )
    .xfail(
        variant_name="sum",
        dtypes=(torch.bfloat16,),
        reason="onnxruntime does not support ml_dtypes.bfloat16",
    ),
    TorchLibOpInfo(
        "ops.aten.slice_scatter",
        core_ops.aten_slice_scatter,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "ops.aten.scatter.src",
        core_ops.aten_scatter_src,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo(
        "ops.aten.scatter.value",
        core_ops.aten_scatter_value,
        supported_dtypes=_TENSOR_DTYPES,
    ),
    TorchLibOpInfo("slice", core_ops.aten_slice, supported_dtypes=_TENSOR_DTYPES),
    TorchLibOpInfo(
        "slice",
        core_ops.aten_slice_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo("sum", core_ops.aten_sum_dim_IntList, supported_dtypes=_REAL_DTYPES),
    TorchLibOpInfo(
        "sum",
        core_ops.aten_sum_dim_IntList_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
    TorchLibOpInfo(
        "transpose", core_ops.aten_transpose, supported_dtypes=_TENSOR_DTYPES
    ),
    TorchLibOpInfo(
        "transpose",
        core_ops.aten_transpose_complex,
        complex=True,
        supported_dtypes=_COMPLEX_DTYPES,
    ),
)


# Source-supported dtypes are test metadata, not exporter dispatch constraints.
TESTED_TORCHLIB_OPS += (
    (
        TorchLibOpInfo(
            "addbmm",
            linalg_ops.aten_addbmm,
            tolerance={torch.float32: (2e-5, 2e-5), torch.float16: (2e-1, 2e-2)},
            supported_dtypes=_REAL_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "addmm",
            linalg_ops.aten_addmm,
            supported_dtypes=_REAL_DTYPES,
        )
        .xfail(
            dtypes=(torch.int16, torch.int32, torch.int64),
            reason="ONNX Runtime does not support int inputs to Gemm",
        )
        .xfail(
            "decomposed",
            dtypes=(torch.int16, torch.int32, torch.int64),
            reason="ONNX Runtime does not support int inputs to Gemm",
        )
        .skip(
            "decomposed",
            matcher=lambda sample: (
                torch.numel(sample.input) == 0
                or torch.numel(sample.args[0]) == 0
                or torch.numel(sample.args[1]) == 0
            ),
            reason="zero sized inputs cannot be compared",
        )
    ),
    (
        TorchLibOpInfo(
            "addmv",
            linalg_ops.aten_addmv,
            tolerance={torch.float16: (2e-3, 2e-2)},
            supported_dtypes=_REAL_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "addr",
            linalg_ops.aten_addr,
            tolerance={torch.float16: (3e-3, 4e-3)},
            supported_dtypes=_REAL_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "atleast_1d_Sequence",
            sequences_ops.aten_atleast_1d_sequence,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
            reason="takes tensor sequences only",
        )
    ),
    (
        TorchLibOpInfo(
            "atleast_2d_Sequence",
            sequences_ops.aten_atleast_2d_sequence,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
            reason="takes tensor sequences only",
        )
    ),
    (
        TorchLibOpInfo(
            "atleast_3d_Sequence",
            sequences_ops.aten_atleast_3d_sequence,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
            reason="takes tensor sequences only",
        )
    ),
    (
        TorchLibOpInfo(
            "baddbmm",
            linalg_ops.aten_baddbmm,
            tolerance={torch.float16: (1e-3, 1e-2)},
            supported_dtypes=_REAL_UINT8_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "bernoulli",
            factories_ops.aten_bernoulli,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            # This string is a unique ID. In extra_opinfo.py, we
            # also define test data for this ID with
            # `opinfo_core.OpInfo("aten.bernoulli.p", ...)`.
            "ops.aten.bernoulli.p",
            factories_ops.aten_bernoulli_p,
            # Skip comparison for the output of this op because it is a random tensor.
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.bernoulli.p_deterministic",
            factories_ops.aten_bernoulli_p,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.blackman_window",
            factories_ops.aten_blackman_window,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "bmm",
            linalg_ops.aten_bmm,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "cat",
            sequences_ops.aten_cat,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: sample.input[0].equal(
                torch.tensor([]).to(sample.input[0].device)
            ),
            reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
        )
    ),
    (
        TorchLibOpInfo(
            "cat",
            sequences_ops.aten_cat_complex,
            complex=True,
            supported_dtypes=_COMPLEX_DTYPES,
        ).skip(
            matcher=lambda sample: sample.input[0].equal(
                torch.tensor([]).to(sample.input[0].device)
            ),
            reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
        )
    ),
    (
        TorchLibOpInfo(
            "concat",
            sequences_ops.aten_cat,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: sample.input[0].equal(
                torch.tensor([]).to(sample.input[0].device)
            ),
            reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
        )
    ),
    (
        TorchLibOpInfo(
            "concatenate",
            sequences_ops.aten_cat,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: sample.input[0].equal(
                torch.tensor([]).to(sample.input[0].device)
            ),
            reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
        )
    ),
    (
        TorchLibOpInfo(
            "dot",
            linalg_ops.aten_dot,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "empty",
            factories_ops.aten_empty,
            input_wrangler=_empty_input_wrangler,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "einsum",
            linalg_ops.aten_einsum,
            input_wrangler=_einsum_input_wrangler,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .xfail(
            reason="fixme: PyTorch produces int64 output with int32 input",
            dtypes=(torch.int32,),
        )
        .xfail(
            reason="fixme: ONNX shape inference fails: https://github.com/onnx/onnx/issues/5739",
            matcher=lambda sample: sample.args[0] == "...ik, ...j -> ij",
        )
    ),
    (
        TorchLibOpInfo(
            "fill",
            factories_ops.aten_fill,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "full",
            factories_ops.aten_full,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "full_like",
            factories_ops.aten_full_like,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            enabled_if=ops_test_common.IS_MACOS,
            reason="fixme: memory allocation issue on CI",
        )
    ),
    (
        TorchLibOpInfo(
            "lift_fresh_copy",
            linalg_ops.aten_lift_fresh_copy,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "linspace",
            factories_ops.aten_linspace,
            tolerance={torch.float16: (2e-2, 2e-3)},
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "matmul",
            linalg_ops.aten_matmul,
            # Windows requires a more relaxed tolerance
            tolerance={torch.float32: (2e-5, 2e-5), torch.float16: (1e-2, 2e-2)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int32,
                torch.int64,
            ),
        ).skip(
            matcher=lambda sample: torch.numel(sample.input) == 0,
            reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
        )
    ),
    (
        TorchLibOpInfo(
            "matmul",
            linalg_ops.aten_matmul_complex,
            # Windows requires a more relaxed tolerance
            tolerance={torch.complex64: (2e-5, 2e-5), torch.complex32: (1e-2, 1e-2)},
            complex=True,
            supported_dtypes=_COMPLEX_DTYPES,
        ).skip(
            matcher=lambda sample: torch.numel(sample.input) == 0,
            reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
        )
    ),
    (
        TorchLibOpInfo(
            "mm",
            linalg_ops.aten_mm,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int32,
                torch.int64,
            ),
        ).skip(
            matcher=lambda sample: torch.numel(sample.input) == 0,
            reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
        )
    ),
    (
        TorchLibOpInfo(
            "mv",
            linalg_ops.aten_mv,
            tolerance={torch.float16: (3e-2, 1e-2)},
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "new_empty",
            factories_ops.aten_new_empty,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "new_empty_strided",
            factories_ops.aten_new_empty_strided,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "new_full",
            factories_ops.aten_new_full,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "new_ones",
            factories_ops.aten_new_ones,
            supported_dtypes=_REAL_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "new_zeros",
            factories_ops.aten_new_zeros,
            supported_dtypes=_REAL_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "normal",
            factories_ops.aten_normal,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .skip(
            matcher=lambda sample: len(sample.args) > 0
            and not isinstance(sample.args[0], float),
            reason="ORT only accept float type for args[0] 'mean'",
        )
        .xfail(
            variant_name="number_mean",
            reason="This variant does not support dtype as an argument",
            matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.normal.float_Tensor",
            factories_ops.aten_normal_float_tensor,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.normal.Tensor_float",
            factories_ops.aten_normal_tensor_float,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.normal.Tensor_Tensor",
            factories_ops.aten_normal_tensor_tensor,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ones",
            factories_ops.aten_ones,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.rand",
            factories_ops.aten_rand,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.rand_like",
            factories_ops.aten_rand_like,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randint",
            factories_ops.aten_randint,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randint.low",
            factories_ops.aten_randint_low,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randint_like",
            factories_ops.aten_randint_like,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randint_like.low_dtype",
            factories_ops.aten_randint_like_low_dtype,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randn",
            factories_ops.aten_randn,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        ).xfail(dtypes=(torch.float16,), reason="fixme: Shape inference error")
    ),
    (
        TorchLibOpInfo(
            "ops.aten.randn_like",
            factories_ops.aten_randn_like,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "repeat_interleave",
            sequences_ops.aten_repeat_interleave_self_int,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .skip(
            matcher=lambda sample: not isinstance(
                sample.kwargs.get("repeats", None), int
            ),
            reason=("ignore cases when repeasts is a Tensor"),
        )
        .skip(dtypes=(torch.bool,), reason="bool not supported")
        .skip(
            matcher=lambda sample: sample.input.numel() == 0,
            reason="fixme: conversion not implemented when input tensor is empty",
        )
    ),
    (
        TorchLibOpInfo(
            "repeat_interleave",
            sequences_ops.aten_repeat_interleave_Tensor,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .skip(
            matcher=lambda sample: isinstance(sample.kwargs.get("repeats", None), int),
            reason=("ignore cases when repeasts is an int"),
        )
        .skip(dtypes=(torch.bool,), reason="bool not supported")
        .skip(
            matcher=lambda sample: sample.input.numel() == 0,
            reason="fixme: conversion not implemented when input tensor is empty",
        )
    ),
    (
        TorchLibOpInfo(
            "resolve_conj",
            linalg_ops.aten_resolve_conj,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "resolve_neg",
            linalg_ops.aten_resolve_neg,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "split_with_sizes",
            sequences_ops.aten_split_with_sizes,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "split",
            sequences_ops.aten_split,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "stack",
            sequences_ops.aten_stack,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "stack",
            sequences_ops.aten_stack_complex,
            complex=True,
            supported_dtypes=_COMPLEX_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "tile",
            sequences_ops.aten_tile,
            supported_dtypes=_TENSOR_DTYPES,
        ).skip(
            matcher=lambda sample: (
                any(dim == 0 for dim in sample.input.shape) or not sample.input.shape
            ),
            reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
        )
    ),
    (
        TorchLibOpInfo(
            "unbind",
            sequences_ops.aten_unbind,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "zeros",
            factories_ops.aten_zeros,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "arange_start_step",
            factories_ops.aten_arange_start_step,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .skip(
            matcher=lambda sample: len(sample.args) != 2,
            reason="arange_start_step overload takes three arguments (input, start, step)",
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("dtype") is None,
            reason="dtype needs to be specified for non-float tensors",
            dtypes=(torch.float16, torch.int64, torch.int32),
        )
    ),
    (
        TorchLibOpInfo(
            "arange_start",
            factories_ops.aten_arange_start,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .skip(
            matcher=lambda sample: len(sample.args) != 1,
            reason="arange_start overload takes two arguments (input, start)",
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("dtype") is None,
            reason="dtype needs to be specified for non-float tensors",
            dtypes=(torch.float16, torch.int64, torch.int32),
        )
    ),
    (
        TorchLibOpInfo(
            "arange",
            factories_ops.aten_arange,
            supported_dtypes=_TENSOR_DTYPES,
        )
        .xfail(
            dtypes=(torch.int32,),
            reason="fixme: output shape mismatch in edge cases. https://github.com/microsoft/onnxscript/issues/974",
        )
        .skip(
            matcher=lambda sample: len(sample.args) != 0,
            reason="arange overload takes single argument",
        )
        .xfail(
            matcher=lambda sample: sample.kwargs.get("end") is not None,
            reason="arange overload does not support positional 'end' argument",
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("dtype") is None,
            reason="dtype needs to be specified for non-float tensors",
            dtypes=(torch.float16, torch.int64, torch.int32),
        )
    ),
    (
        TorchLibOpInfo(
            "contiguous",
            linalg_ops.aten_contiguous,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "empty_like",
            factories_ops.aten_empty_like,
            nondeterministic=True,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.hamming_window",
            factories_ops.aten_hamming_window,
            tolerance={torch.float32: (8e-2, 6e-3)},
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.hann_window",
            factories_ops.aten_hann_window,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "multinomial",
            factories_ops.aten_multinomial,
            nondeterministic=True,
            supported_dtypes=_FLOAT_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ones_like",
            factories_ops.aten_ones_like,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.tensor.bool",
            factories_ops.aten_tensor_bool,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.tensor.float",
            factories_ops.aten_tensor_float,
            supported_dtypes=_TENSOR_DTYPES,  # Custom from extra_opinfo
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.tensor.int",
            factories_ops.aten_tensor_int,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
    (
        TorchLibOpInfo(
            "zeros_like",
            factories_ops.aten_zeros_like,
            supported_dtypes=_TENSOR_DTYPES,
        )
    ),
)

TESTED_TORCHLIB_OPS += (
    TorchLibOpInfo("chunk", sequences_ops.aten_chunk, supported_dtypes=_TENSOR_DTYPES),
)


def _embedding_bag_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # ONNX attributes cannot be None; omit padding_idx if it's None.
    if "padding_idx" in kwargs:
        padding_idx = kwargs.pop("padding_idx")
        if padding_idx is not None:
            kwargs["padding_idx"] = int(padding_idx)

    # Ensure indices/offsets are int64 (positional: weight, indices, offsets, ...)
    if len(args) >= 3:
        if isinstance(args[1], torch.Tensor):
            args[1] = args[1].to(torch.long)
        if isinstance(args[2], torch.Tensor):
            args[2] = args[2].to(torch.long)

    return args, kwargs


# Source-supported dtypes are test metadata, not exporter dispatch constraints.
TESTED_TORCHLIB_OPS += (
    (
        TorchLibOpInfo(
            "ops.aten._log_softmax",
            residual_ops.aten__log_softmax,
            tolerance={torch.float16: (1e-3, 1e-3)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._softmax",
            residual_ops.aten__softmax,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "allclose",
            residual_ops.aten_allclose,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "bilinear",
            residual_ops.aten_bilinear,
            tolerance={torch.float32: (2e-5, 2e-5)},
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "complex",
            residual_ops.aten_complex,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "constant_pad_nd",
            residual_ops.aten_constant_pad_nd,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "cross",
            residual_ops.aten_cross,
            tolerance={torch.float16: (6e-2, 2e-1)},
            supported_dtypes=(
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
            ),
        ).skip(
            dtypes=(torch.float16 if sys.platform != "linux" else torch.complex64,),
            reason="fixme: test is failing on windows and torch nightly",
        )
    ),
    (
        TorchLibOpInfo(
            "equal",
            residual_ops.aten_equal,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.fake_quantize_per_channel_affine",
            residual_ops.aten_fake_quantize_per_channel_affine,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).xfail(
            reason="fixme: ONNX (De)QuantizeLinear only supports integer zero_point values",
            matcher=lambda sample: sample.args[1].dtype != torch.int32,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.fake_quantize_per_tensor_affine",
            residual_ops.aten_fake_quantize_per_tensor_affine,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._grouped_mm",
            residual_ops.aten_grouped_mm,
            tolerance={torch.float32: (1e-5, 1e-5)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            enabled_if=not hasattr(torch.ops.aten, "_grouped_mm"),
            reason="torch.ops.aten._grouped_mm is not available in this version of PyTorch",
        )
    ),
    (
        TorchLibOpInfo(
            "histc",
            residual_ops.aten_histc,
            supported_dtypes=(
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
            ),
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("min") == sample.kwargs.get("max"),
            reason="then min=sample.min(), max=sample.max(), torch.histc does not "
            "define what happens when both are equal (1 sample with one element "
            "for example). torch does something, maybe "
            "something like zeros(bins)[bins // 2 + 1] = 1.",
        )
        .skip(
            dtypes=(torch.float16,),
            reason="we skip float16 because of https://github.com/pytorch/pytorch/issues/174668",
        )
    ),
    (
        TorchLibOpInfo(
            "logcumsumexp",
            residual_ops.aten_logcumsumexp,
            tolerance={torch.float16: (1e-2, 1e-1)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "logdet",
            residual_ops.aten_logdet,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.native_dropout",
            residual_ops.aten_native_dropout,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.dropout",
            residual_ops.aten_dropout,
            input_wrangler=_dropout_input_wrangler,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            matcher=lambda sample: len(sample.kwargs) == 0
            or sample.kwargs.get("p", 0.0) > 0.0,
            reason="dropout is random so the result not match",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.embedding_bag",
            embedding_ops.aten_embedding_bag,
            tolerance={torch.float32: (1e-4, 5e-4)},
            compare_shape_only_for_output=(1, 2, 3),
            input_wrangler=_embedding_bag_input_wrangler,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            dtypes=(torch.float16,),
            reason="fixme: results mismatch in torch nightly.",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.embedding_bag.padding_idx",
            embedding_ops.aten_embedding_bag_padding_idx,
            tolerance={torch.float16: (1e-2, 1e-2)},
            compare_shape_only_for_output=(1, 2, 3),
            input_wrangler=_embedding_bag_input_wrangler,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.embedding_renorm",
            embedding_ops.aten_embedding_renorm,
            tolerance={torch.float16: (1e-2, 1e-2)},
            compare_shape_only_for_output=(1, 2, 3),
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.embedding",
            embedding_ops.aten_embedding,
            input_wrangler=_embedding_input_wrangler,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.pixel_shuffle",
            residual_ops.aten_pixel_shuffle,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        ).xfail(
            dtypes=(torch.int32, torch.int64),
            reason="fixme: ONNX Runtime does not support int32/64 inputs",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.pixel_unshuffle",
            residual_ops.aten_pixel_unshuffle,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        )
        .skip(
            matcher=lambda sample: sample.input.numel() == 0,
            reason="fixme: size 0 inputs are not handled yet",
        )
        .xfail(
            dtypes=(torch.int32, torch.int64),
            reason="fixme: ONNX Runtime does not support int32/64 inputs",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.selu",
            residual_ops.aten_selu,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "polar",
            residual_ops.aten_polar,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.prelu",
            residual_ops.aten_prelu,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "sigmoid",
            residual_ops.aten_sigmoid,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "signbit",
            residual_ops.aten_signbit,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "softmax",
            residual_ops.aten_softmax,
            tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (3e-4, 4e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
        .xfail(
            dtypes=(torch.float16,),
            reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
            test_class_name="TestOutputConsistencyFullGraph",
        )
        .xfail(
            variant_name="with_dtype",
            dtypes=(torch.float16,),
            reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
            test_class_name="TestOutputConsistencyFullGraph",
        )
        .skip(
            matcher=lambda sample: len(sample.input.shape) == 0,
            reason="fixme: SoftMax does not support empty tensor as input",
        )
        .skip(
            variant_name="with_dtype",
            matcher=lambda sample: len(sample.input.shape) == 0,
            reason="fixme: SoftMax does not support empty tensor as input",
        )
    ),
    (
        TorchLibOpInfo(
            "tril",
            residual_ops.aten_tril,
            supported_dtypes=(
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
            ),
        ).xfail(
            dtypes=(torch.int32,),
            reason="fixme: ORT does not have an implementation of Trilu for int32.",
        )
    ),
    (
        TorchLibOpInfo(
            "triu",
            residual_ops.aten_triu,
            supported_dtypes=(
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
            ),
        ).xfail(
            dtypes=(torch.int32,),
            reason="fixme: ORT does not have an implementation of Trilu for int32.",
        )
    ),
    (
        TorchLibOpInfo(
            "grid_sampler_2d",
            residual_ops.aten_grid_sampler_2d,
            supported_dtypes=(
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
            ),
        )
        .skip(
            # Torch implemented this using the cubic convolution algorithm with alpha=-0.75, might be different than ORT
            matcher=lambda sample: sample.args[1] == 2,
            reason="fixme: 'bicubic' mode in ORT implemented differently with Torch",
        )
        .skip(dtypes=(torch.float16,), reason="fixme: Accuracy is not high enough")
    ),
    (
        TorchLibOpInfo(
            "heaviside",
            residual_ops.aten_heaviside,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.grid_sample",
            residual_ops.aten_grid_sampler,
            input_wrangler=_grid_sample_input_wrangler,
            tolerance={torch.float16: (8e-2, 2e-3)},
            supported_dtypes=(
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
            ),
        ).skip(
            # Torch implemented this using the cubic convolution algorithm with alpha=-0.75, might be different than ORT
            matcher=lambda sample: (
                sample.kwargs.get("mode") == "bicubic" or len(sample.args[0].shape) != 4
            ),
            reason="fixme: 'bicubic' mode in ORT implemented differently with Torch and only support 4D-tensor",
        )
    ),
    (
        TorchLibOpInfo(
            "roll",
            residual_ops.aten_roll,
            input_wrangler=_roll_input_wrangler,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "roll",
            residual_ops.aten_roll_complex,
            input_wrangler=_roll_input_wrangler,
            complex=True,
            supported_dtypes=(
                torch.complex32,
                torch.complex64,
                torch.complex128,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.stft",  # Custom from extra_opinfo
            residual_ops.aten_stft,
            tolerance={torch.float32: (3.7e-5, 1.8e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).xfail(
            dtypes=(torch.float16,),
            reason="RuntimeError: MKL FFT doesn't support tensors of type: Half",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._unique.default",
            residual_ops.aten__unique,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._unique2.default",
            residual_ops.aten__unique2,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.unique_dim.default",
            residual_ops.aten_unique_dim,
            supported_dtypes=(
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
            ),
        ).skip(
            device_type="cpu",
            reason=(
                "ops.aten.unique_dim.default returns different shapes for optional outputs on CPU/CUDA. "
                "Our implementation is based on that for CUDA"
            ),
        )
    ),
)


_NN_DTYPES_0 = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
_NN_DTYPES_1 = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_NN_DTYPES_2 = (
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
_NN_DTYPES_3 = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.uint8,
)
_NN_DTYPES_4 = (torch.complex64, torch.complex128)

# Source dtype eligibility is numerical test metadata, not exporter dispatch.
TESTED_TORCHLIB_OPS += (
    (
        TorchLibOpInfo(
            "nn.functional.celu",
            neural_ops.aten_celu,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.cross_entropy",
            # use cross_entropy as test case instead of cross_entropy_loss (not in OPS_DB)
            neural_ops.aten_cross_entropy_loss,
            tolerance={torch.float16: (1e-2, 1e-2)},
            input_wrangler=_cross_entropy_input_wrangler,
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            matcher=lambda sample: (
                len(sample.args) < 1
                or (
                    isinstance(sample.args[0], torch.Tensor)
                    and sample.args[0].dtype != torch.int64
                )
            ),
            reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[target] as int type",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.elu",
            neural_ops.aten_elu,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.hardsigmoid",
            neural_ops.aten_hardsigmoid,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.hardswish",
            neural_ops.aten_hardswish,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.hardtanh",
            neural_ops.aten_hardtanh,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.leaky_relu",
            neural_ops.aten_leaky_relu,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.logsigmoid",
            neural_ops.aten_log_sigmoid,
            tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (8e-2, 4e-4)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.mish",
            neural_ops.aten_mish,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.nll_loss",
            neural_ops.aten_nll_loss,
            input_wrangler=_nll_loss_input_wrangler,
            tolerance={torch.float16: (5e-2, 1e-2)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.pad",
            neural_ops.aten_pad,
            supported_dtypes=_NN_DTYPES_2,
        )
        .skip(
            variant_name="circular",
            reason="fixme: ORT does not support the circular mode",
        )
        .skip(
            variant_name="replicate_negative",
            reason="fixme: The implementation for negative paddings is not correct",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.reflection_pad1d",
            neural_ops.aten_reflection_pad1d,
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            dtypes=(torch.int64,),
            reason="Torch not implement reflection_pad1d for int64.",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.reflection_pad2d",
            neural_ops.aten_reflection_pad2d,
            input_wrangler=_reflection_pad2d_input_wrangler,
            supported_dtypes=_NN_DTYPES_2,
        ).skip(
            matcher=lambda sample: not (
                len(sample.args) > 1 and sample.args[1] == "reflect"
            ),
            reason="this Aten overload need args[1] == 'reflect' for pad mode",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.relu",
            neural_ops.aten_relu,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.relu6",
            neural_ops.aten_relu6,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.replication_pad1d",
            neural_ops.aten_replication_pad1d,
            supported_dtypes=_NN_DTYPES_2,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.replication_pad2d",
            neural_ops.aten_replication_pad2d,
            input_wrangler=_replication_pad2d_input_wrangler,
            supported_dtypes=_NN_DTYPES_2,
        )
        .skip(
            matcher=lambda sample: not (
                len(sample.args) > 1 and sample.args[1] == "replicate"
            ),
            reason="this Aten overload need args[1] == 'replicate' for pad mode",
        )
        .skip(
            variant_name="replicate_negative",
            reason="fixme: The implementation for negative paddings is not correct. Potentially an ORT issue",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.replication_pad3d",
            neural_ops.aten_replication_pad3d,
            input_wrangler=_replication_pad3d_input_wrangler,
            supported_dtypes=_NN_DTYPES_2,
        ).skip(
            matcher=lambda sample: (
                not (
                    len(sample.args) > 1
                    and sample.args[1] == "replicate"
                    and len(sample.input.shape) == 5
                )
            ),
            reason="this Aten overload need args[1] == 'replicate' for pad mode, and 3D tensor",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.mse_loss",
            neural_ops.aten_mse_loss,
            input_wrangler=_mse_loss_input_wrangler,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.silu",
            neural_ops.aten_silu,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.softplus",
            neural_ops.aten_softplus,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.col2im",
            neural_ops.aten_col2im,
            supported_dtypes=_NN_DTYPES_1,
        ).xfail(
            dtypes=(torch.float16,),
            reason="fixme: Tensor-likes are not close. https://github.com/microsoft/onnxruntime/issues/16007",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.convolution",
            normalization_ops.aten_convolution,
            tolerance={torch.float32: (2e-4, 9e-4)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "group_norm_op18",
            neural_ops.aten_group_norm,
            tolerance={torch.float16: (1e-2, 7e-3)},
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
            reason="Using op.InstanceNormalization to simulate GroupNorm, which does not support 0-dim input",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.layer_norm",
            normalization_ops.aten_layer_norm,
            tolerance={torch.float32: (3.7e-5, 1.8e-4)},
            supported_dtypes=_NN_DTYPES_1,
        )
        .xfail(
            dtypes=(torch.int64,),
            reason="fixme: ORT `LayerNormKernelImpl` not implemented for int64",
        )
        .skip(
            matcher=lambda sample: sample.input.shape[-1] <= 1,
            reason="fixme: onnxruntime fail when no reduction is needed",
        )
        .skip(
            dtypes=(torch.float32 if sys.platform != "linux" else torch.complex64,),
            reason="fixme: test is unstable on macosx, windows",
        )
    ),
    (
        TorchLibOpInfo(
            # Custom from extra_opinfo
            "ops.aten.max_pool1d",
            neural_ops.aten_max_pool1d,
            supported_dtypes=_NN_DTYPES_3,
        )
    ),
    (
        TorchLibOpInfo(
            # Custom from extra_opinfo
            "ops.aten.max_pool2d",
            neural_ops.aten_max_pool2d,
            supported_dtypes=_NN_DTYPES_3,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.max_pool3d",  # Custom from extra_opinfo
            neural_ops.aten_max_pool3d,
            supported_dtypes=_NN_DTYPES_3,
        ).xfail(
            variant_name="empty_strides",
            reason="fixme: 'shape' do not match: torch.Size([2, 3, 4, 3]) != torch.Size([2, 3, 4, 2]). https://github.com/microsoft/onnxscript/issues/975",
        )
    ),
    (
        TorchLibOpInfo(
            "native_batch_norm",
            normalization_ops.aten_native_batch_norm,
            tolerance={torch.float16: (1e-2, 7e-3)},
            supported_dtypes=_NN_DTYPES_0,
        )
        .skip(
            device_type="cpu",
            matcher=lambda sample: sample.args[-3] is False,
            reason="native_batch_norm outputs different shapes on CPU and CUDA when training is False. Our implematation is based on that for CUDA",
        )
        .skip(
            device_type="cpu",
            dtypes=(torch.float16,),
            reason="native_batch_norm outputs different dtypes on CPU and CUDA. Our implematation is based on that for CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._native_batch_norm_legit",
            normalization_ops.aten_native_batch_norm,
            tolerance={torch.float16: (1e-2, 7e-3)},
            supported_dtypes=_NN_DTYPES_0,
        ).skip(
            device_type="cpu",
            matcher=lambda sample: sample.kwargs.get("training") is False,
            reason="native_batch_norm outputs different shapes on CPU and CUDA when training is False. Our implematation is based on that for CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._native_batch_norm_legit.no_stats",
            normalization_ops.aten__native_batch_norm_no_stats,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._native_batch_norm_legit_functional",
            normalization_ops.aten__native_batch_norm_legit_functional,
            tolerance={torch.float16: (1e-2, 7e-3)},
            supported_dtypes=_NN_DTYPES_0,
        ).skip(
            device_type="cpu",
            matcher=lambda sample: sample.kwargs.get("training") is False,
            reason="native_batch_norm outputs different results on CPU and CUDA when training is False. Our implematation is based on that for CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.native_group_norm",
            normalization_ops.aten_native_group_norm,
            tolerance={torch.float16: (1e-2, 7e-3)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "native_layer_norm",
            normalization_ops.aten_native_layer_norm,
            tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (1e-1, 7e-4)},
            supported_dtypes=_NN_DTYPES_1,
        )
        .skip(
            dtypes=(torch.float32,),
            matcher=lambda sample: sample.input.shape[-1] <= 1,
            # enabled_if=ops_test_common.IS_MACOS,
            reason="fixme: result mismatch. https://github.com/microsoft/onnxruntime/issues/20676",
        )
        .skip(
            dtypes=(torch.float16,),
            device_type="cpu",
            reason="native_layer_norm outputs different dtypes on CPU and CUDA. Our implematation is based on that for CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.avg_pool1d",
            neural_ops.aten_avg_pool1d,
            input_wrangler=_avg_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_0,
        )
        .xfail(
            matcher=lambda sample: (
                (len(sample.args) > 5 and sample.args[5] is not None)
                or (sample.kwargs.get("divisor_override") is not None)
            ),
            reason="ONNX doesn't support divisor_override argument",
        )
        .xfail(
            matcher=lambda sample: (
                (sample.kwargs.get("ceil_mode") is True)
                and (
                    sample.kwargs.get("count_include_pad") is True
                    or sample.input.shape[2]
                    % (
                        sample.args[0][0]
                        if isinstance(sample.args[0], tuple)
                        else sample.args[0]
                    )
                    != 0
                )
            ),
            reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.avg_pool2d",
            neural_ops.aten_avg_pool2d,
            input_wrangler=_avg_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            matcher=lambda sample: (
                (len(sample.args) > 5 and sample.args[5] is not None)
                or (sample.kwargs.get("divisor_override") is not None)
            ),
            reason="ONNX doesn't support divisor_override argument",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.avg_pool3d",
            neural_ops.aten_avg_pool3d,
            input_wrangler=_avg_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_0,
        )
        .xfail(
            matcher=lambda sample: (
                (len(sample.args) > 5 and sample.args[5] is not None)
                or (sample.kwargs.get("divisor_override") is not None)
            ),
            reason="ONNX doesn't support divisor_override argument",
        )
        .xfail(
            matcher=lambda sample: sample.kwargs.get("ceil_mode") is True,
            reason="fixme(after opset19): ORT doesn't match PyTorch when ceil_mode=True until opset 19",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.conv1d",
            normalization_ops.aten_conv1d,
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
            reason="String padding is not accepted by aten::conv1d",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.conv1d",
            normalization_ops.aten_conv1d_complex,
            complex=True,
            tolerance={torch.complex64: (2e-5, 3e-5)},
            supported_dtypes=_NN_DTYPES_4,
        ).xfail(
            matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
            reason="String padding is not accepted by aten::conv1d",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.conv2d",
            normalization_ops.aten_conv2d,
            tolerance={torch.float32: (2e-5, 3e-5)},
            supported_dtypes=_NN_DTYPES_0,
        ).xfail(
            matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
            reason="String padding is not accepted by aten::conv2d",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.conv2d",
            normalization_ops.aten_conv2d_complex,
            complex=True,
            tolerance={torch.complex64: (4e-5, 6e-5)},
            supported_dtypes=_NN_DTYPES_4,
        ).xfail(
            matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
            reason="String padding is not accepted by aten::conv2d",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.instance_norm",
            normalization_ops.aten_instance_norm,
            tolerance={torch.float16: (1e-2, 1e-3)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.conv3d",
            normalization_ops.aten_conv3d,
            tolerance={torch.float32: (3.7e-5, 1.8e-4)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.conv3d",
            normalization_ops.aten_conv3d_complex,
            complex=True,
            tolerance={torch.complex64: (1e-4, 5e-4)},
            supported_dtypes=_NN_DTYPES_4,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.gelu",
            neural_ops.aten_gelu,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.glu",
            neural_ops.aten_glu,
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.linear",
            neural_ops.aten_linear,
            tolerance={torch.float16: (1e-2, 1e-3)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.linear.1d_weight",
            neural_ops.aten_linear,
            tolerance={torch.float16: (1e-2, 1e-3)},
            supported_dtypes=_NN_DTYPES_0,
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.unfold",
            neural_ops.aten_im2col,
            input_wrangler=_im2col_input_wrangler,
            supported_dtypes=_NN_DTYPES_1,
        ).xfail(
            matcher=lambda sample: (
                any(dim == 0 for dim in sample.input.shape) or not sample.input.shape
            ),
            reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool1d",
            neural_ops.aten_max_pool1d,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        ).skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is True,
            reason="this aten overload assume return_indices=False",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool1d_with_indices",
            neural_ops.aten_max_pool1d_with_indices,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        ).skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is False,
            reason="this aten overload assume return_indices=True",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool2d",
            neural_ops.aten_max_pool2d,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        ).skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is True,
            reason="this aten overload assume return_indices=False",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool2d_with_indices",
            neural_ops.aten_max_pool2d_with_indices,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        ).skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is False,
            reason="this aten overload assume return_indices=True",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool3d",
            neural_ops.aten_max_pool3d,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        )
        .skip(
            matcher=lambda sample: (
                sample.kwargs.get("ceil_mode") is True
                and sample.kwargs.get("padding") == 1
            ),
            reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is True,
            reason="this aten overload assume return_indices=False",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.max_pool3d_with_indices",
            neural_ops.aten_max_pool3d_with_indices,
            input_wrangler=_max_pool_input_wrangler,
            supported_dtypes=_NN_DTYPES_3,
        )
        .skip(
            matcher=lambda sample: (
                sample.kwargs.get("ceil_mode") is True
                and sample.kwargs.get("padding") == 1
            ),
            reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("return_indices") is False,
            reason="this aten overload assume return_indices=True",
        )
    ),
    (
        TorchLibOpInfo(
            "nn.functional.scaled_dot_product_attention",
            neural_ops.aten_scaled_dot_product_attention,
            tolerance={torch.float32: (3e-4, 1.5e-5)},
            supported_dtypes=_NN_DTYPES_0,
        )
        .skip(
            matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
            reason="dropout is random so the results do not match",
        )
        .xfail(
            dtypes=(torch.float16,),
            reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
            test_class_name="TestOutputConsistencyFullGraph",
        )
        .xfail(
            matcher=lambda sample: (
                len(sample.input.shape) != 4
                or len(sample.args[0].shape) != 4
                or len(sample.args[1].shape) != 4
            ),
            reason="torch sdpa is expected to pass in 4d q, k, and v.",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_bilinear2d.default",
            neural_ops.aten_upsample_bilinear2d,
            supported_dtypes=_NN_DTYPES_1,
        ).xfail(
            matcher=lambda sample: (
                sample.args[1] is False and sample.kwargs.get("scales_h") is not None
            ),
            reason="fixme: align_corners=False output mismatch when scales are provided",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._upsample_bilinear2d_aa",
            neural_ops.aten__upsample_bilinear2d_aa,
            supported_dtypes=_NN_DTYPES_1,
        )
        .xfail(
            matcher=lambda sample: sample.args[1] is True,
            reason="fixme: align_corners=True diverges between PyTorch AA kernel and ONNX Resize antialias",
        )
        .xfail(
            matcher=lambda sample: (
                sample.args[1] is False and sample.kwargs.get("scales_h") is not None
            ),
            reason="fixme: align_corners=False output mismatch when scales are provided",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_bilinear2d.vec",
            neural_ops.aten_upsample_bilinear2d_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_bicubic2d.default",
            neural_ops.aten_upsample_bicubic2d,
            supported_dtypes=_NN_DTYPES_1,
        ).xfail(
            matcher=lambda sample: (
                sample.args[1] is False and sample.kwargs.get("scales_h") is not None
            ),
            reason="fixme: align_corners=False output mismatch when scales are provided",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._upsample_bicubic2d_aa",
            neural_ops.aten__upsample_bicubic2d_aa,
            supported_dtypes=_NN_DTYPES_1,
        )
        .xfail(
            matcher=lambda sample: sample.args[1] is True,
            reason="fixme: align_corners=True diverges between PyTorch AA kernel and ONNX Resize antialias",
        )
        .xfail(
            matcher=lambda sample: (
                sample.args[1] is False and sample.kwargs.get("scales_h") is not None
            ),
            reason="fixme: align_corners=False output mismatch when scales are provided",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_bicubic2d.vec",
            neural_ops.aten_upsample_bicubic2d_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_linear1d",
            neural_ops.aten_upsample_linear1d,
            supported_dtypes=_NN_DTYPES_1,
        ).xfail(
            matcher=lambda sample: (
                sample.args[1] is False and sample.kwargs.get("scales") is not None
            ),
            reason="fixme: align_corners=False output mismatch when scales are provided",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_linear1d.vec",
            neural_ops.aten_upsample_linear1d_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest1d",
            neural_ops.aten_upsample_nearest1d,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest1d.vec",
            neural_ops.aten_upsample_nearestnd_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest2d",
            neural_ops.aten_upsample_nearest2d,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest2d.vec",
            neural_ops.aten_upsample_nearestnd_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest3d",
            neural_ops.aten_upsample_nearest3d,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_nearest3d.vec",
            neural_ops.aten_upsample_nearestnd_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_trilinear3d.default",
            neural_ops.aten_upsample_trilinear3d,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten.upsample_trilinear3d.vec",
            neural_ops.aten_upsample_trilinear3d_vec,
            supported_dtypes=_NN_DTYPES_1,
        )
    ),
)

ops_test_common.duplicate_opinfo(
    OPS_DB, "nn.functional.group_norm", ("group_norm_op18",)
)


# Source dtype eligibility is numerical test metadata, not exporter dispatch.
TESTED_TORCHLIB_OPS += (
    (
        TorchLibOpInfo(
            "ops.aten._fft_c2c",  # Custom from extra_opinfo
            fft_ops.aten__fft_c2c,
            tolerance={torch.complex64: (3e-3, 1.8e-4)},
            complex=True,
            supported_dtypes=(
                torch.complex64,
                torch.complex128,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._fft_c2r",  # Custom from extra_opinfo
            fft_ops.aten__fft_c2r,
            tolerance={torch.complex64: (3e-3, 1.8e-4)},
            complex=True,
            supported_dtypes=(
                torch.complex64,
                torch.complex128,
            ),
        ).xfail(
            matcher=lambda sample: True,
            reason="Requires ONNX with IRFFT support (onesided=True, inverse=True)",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._fft_r2c",  # Custom from extra_opinfo
            fft_ops.aten__fft_r2c,
            tolerance={torch.float64: (2e-6, 2e-6), torch.float32: (3e-2, 3e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "erf",
            special_ops.aten_special_erf,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "erfc",
            special_ops.aten_special_erfc,
            tolerance={torch.float16: (5e-1, 2e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "expm1",
            special_ops.aten_special_expm1,
            tolerance={torch.float16: (1e-2, 2e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "special.erfcx",
            special_ops.aten_special_erfcx,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).xfail(
            reason="fixme: The implementation is numerically unstable: https://github.com/microsoft/onnxscript/issues/1223"
        )
    ),
    (
        TorchLibOpInfo(
            "linalg.det",
            linalg_ops.aten_linalg_det,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "linalg.vector_norm",
            linalg_ops.aten_linalg_vector_norm,
            tolerance={torch.float16: (2e-3, 2e-3)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            matcher=lambda sample: sample.kwargs.get("ord") == 6,
            dtypes=(torch.float16,),
            reason="ORT returns a more accurate value for float16 with ord=6 (expected=Inf, actual=9.48).",
        )
    ),
    (
        TorchLibOpInfo(
            "log_softmax",
            special_ops.aten_special_log_softmax,
            tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (4e-4, 6e-3)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
        .xfail(
            dtypes=(torch.float16,),
            reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
            test_class_name="TestOutputConsistencyFullGraph",
        )
        .xfail(
            variant_name="with_dtype",
            dtypes=(torch.float16,),
            reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
            test_class_name="TestOutputConsistencyFullGraph",
        )
        .skip(
            matcher=lambda sample: len(sample.input.shape) == 0,
            reason="fixme: LogSoftMax does not support empty tensor as input",
        )
        .skip(
            variant_name="with_dtype",
            matcher=lambda sample: len(sample.input.shape) == 0,
            reason="fixme: LogSoftMax does not support empty tensor as input",
        )
    ),
    (
        TorchLibOpInfo(
            "sinc",
            special_ops.aten_special_sinc,
            tolerance={torch.float16: (1e-2, 6e-4)},
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "xlogy",
            special_ops.aten_special_xlogy,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._scaled_dot_product_flash_attention",
            neural_ops.aten__scaled_dot_product_flash_attention,
            tolerance={torch.float32: (3e-4, 1.5e-5)},
            # Output[0] is OK, but other outputs just have the same shape with zero values
            nondeterministic=True,
            compare_shape_only_for_output=(1, 2, 3, 4, 5, 6, 7, 8),
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            device_type="cpu",
            reason="_scaled_dot_product_flash_attention only supports CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.aten._scaled_dot_product_efficient_attention",
            neural_ops.aten__scaled_dot_product_efficient_attention,
            tolerance={torch.float32: (3e-4, 1.5e-5)},
            # Output[0] is OK, but other outputs just have the same shape with zero values
            nondeterministic=True,
            compare_shape_only_for_output=(1, 2, 3),
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        ).skip(
            enabled_if=not torch.cuda.is_available(),
            reason="_scaled_dot_product_efficient_attention only supports CUDA",
        )
    ),
    (
        TorchLibOpInfo(
            "ops.prims.broadcast_in_dim.default",
            prims_ops.prims_broadcast_in_dim,
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "ops.prims.var.default",
            prims_ops.prims_var,
            tolerance={torch.float16: (1e-3, 5e-2)},
            supported_dtypes=(
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
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "torchvision.ops.nms",
            vision_ops.torchvision_nms,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "torchvision.ops.roi_align",
            vision_ops.torchvision_roi_align,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
    ),
    (
        TorchLibOpInfo(
            "torchvision.ops.roi_pool",
            vision_ops.torchvision_roi_pool,
            supported_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
        )
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
