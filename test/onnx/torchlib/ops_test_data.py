# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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
from typing import Any, Callable, Collection, Optional
from typing_extensions import Self

import extra_opinfo
import numpy as np
import ops_test_common

import torch
from torch.onnx._internal.exporter._torchlib.ops import (
    core as core_ops,
    fft as fft_ops,
    linalg as linalg_ops,
    nn as nn_ops,
    prims as prims_ops,
    special as special_ops,
)
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import definitions as opinfo_definitions


# Create a copy of the op_db to modify
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

# Append extra op_db into the op database for testing
OPS_DB.extend(opinfo_definitions.signal.op_db)
OPS_DB.extend(extra_opinfo.OP_DB)


@dataclasses.dataclass
class TorchLibOpInfo:
    """A dataclass to store the information to test an torchlib op."""

    # The name of the op_info, e.g. "add"
    op_info_name: str
    # The torchlib ONNX Function to test
    op: Callable[..., Any]
    # The input wrangler function to adjust the input to fit the aten signature
    input_wrangler: Optional[
        Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]]
    ] = None
    # Whether the op is non-deterministic
    nondeterministic: bool = False
    # Whether to compare the shape only for the output[index]
    # For example: (1,2) means compare value for output[0] and shape for output[1] and [2]
    # We may be able to combine this with the nondeterministic option
    compare_shape_only_for_output: tuple[int, ...] = ()
    # Whether the function is designed for complex inputs
    complex: bool = False
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
        dtypes: Optional[Collection[torch.dtype]] = None,
        device_type: Optional[str] = None,
        matcher: Optional[Callable[[Any], Any]] = None,
        enabled_if: bool = True,
        test_class_name: Optional[str] = None,
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
        dtypes: Optional[Collection[torch.dtype]] = None,
        device_type: Optional[str] = None,
        matcher: Optional[Callable[[Any], Any]] = None,
        enabled_if: bool = True,
        test_class_name: Optional[str] = None,
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
    # Convert string attriute to int as input
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


def _linalg_vector_norm_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the dims as tensor
    if "dim" in kwargs:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
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


def _permute_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Change the dims argument back to a list because ONNX Transpose does not
    # support dynamic perms
    kwargs["dims"] = args.pop()
    kwargs["dims"] = kwargs["dims"].tolist()
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
        if isinstance(args[1], int):  # convert shift to tensor
            args[1] = np.array([args[1]], dtype=np.int64)
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
    TorchLibOpInfo(
        "ops.aten._fft_c2c",  # Custom from extra_opinfo
        fft_ops.aten__fft_c2c,
        tolerance={torch.complex64: (3e-3, 1.8e-4)},
        complex=True,
    ),
    TorchLibOpInfo(
        "ops.aten._fft_c2r",  # Custom from extra_opinfo
        fft_ops.aten__fft_c2r,
        tolerance={torch.complex64: (3e-3, 1.8e-4)},
        complex=True,
    ).xfail(
        dtypes=(torch.complex64,),
        reason="fixme: the result is wrong: https://github.com/microsoft/onnxscript/pull/926",
    ),
    TorchLibOpInfo(
        "ops.aten._fft_r2c",  # Custom from extra_opinfo
        fft_ops.aten__fft_r2c,
        tolerance={torch.float64: (2e-6, 2e-6), torch.float32: (3e-2, 3e-4)},
    ),
    TorchLibOpInfo(
        "ops.aten._local_scalar_dense",
        core_ops.aten__local_scalar_dense,
    ),
    TorchLibOpInfo("ops.aten._log_softmax", core_ops.aten__log_softmax),
    TorchLibOpInfo(
        "ops.aten._log_softmax_half",
        core_ops.aten__log_softmax_half,
        tolerance={torch.float16: (1e-3, 1e-3)},
    ),
    TorchLibOpInfo("ops.aten._softmax", core_ops.aten__softmax),
    TorchLibOpInfo("ops.aten._softmax_half", core_ops.aten__softmax_half),
    TorchLibOpInfo("all_dim", core_ops.aten_all_dim).skip(
        matcher=lambda sample: not (len(sample.kwargs) > 0)
        or isinstance(sample.kwargs.get("dim"), tuple),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design. dim must be an integer",
    ),
    TorchLibOpInfo("all_dims", core_ops.aten_all_dims).skip(
        matcher=lambda sample: not isinstance(sample.kwargs.get("dim"), tuple),
        reason="this overload requires dim to be a tuple",
    ),
    TorchLibOpInfo("allclose", core_ops.aten_allclose),
    TorchLibOpInfo(
        "all",
        core_ops.aten_all,
    ).skip(
        matcher=lambda sample: len(sample.kwargs) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo("abs", core_ops.aten_abs),
    TorchLibOpInfo("abs", core_ops.aten_abs_complex, complex=True),
    TorchLibOpInfo("acos", core_ops.aten_acos),
    TorchLibOpInfo("acosh", core_ops.aten_acosh),
    TorchLibOpInfo("add", core_ops.aten_add, tolerance={torch.float16: (1e-3, 1e-3)}),
    TorchLibOpInfo("add", core_ops.aten_add_complex, complex=True),
    TorchLibOpInfo(
        "addbmm",
        core_ops.aten_addbmm,
        tolerance={torch.float32: (2e-5, 2e-5), torch.float16: (2e-1, 2e-2)},
    ),
    TorchLibOpInfo(
        "addcdiv", core_ops.aten_addcdiv, tolerance={torch.float16: (3e-2, 1e-3)}
    ),
    TorchLibOpInfo(
        "addcmul", core_ops.aten_addcmul, tolerance={torch.float16: (4e-3, 3e-3)}
    ),
    TorchLibOpInfo("addmm", core_ops.aten_addmm)
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
        matcher=lambda sample: torch.numel(sample.input) == 0
        or torch.numel(sample.args[0]) == 0
        or torch.numel(sample.args[1]) == 0,
        reason="zero sized inputs cannot be compared",
    ),
    TorchLibOpInfo(
        "addmv", core_ops.aten_addmv, tolerance={torch.float16: (2e-3, 2e-2)}
    ),
    TorchLibOpInfo(
        "addr",
        core_ops.aten_addr,
        tolerance={torch.float16: (3e-3, 4e-3)},
    ),
    TorchLibOpInfo(
        "amax",
        core_ops.aten_amax,
        input_wrangler=_amin_amax_input_wrangler,
    ),
    TorchLibOpInfo(
        "amin",
        core_ops.aten_amin,
        input_wrangler=_amin_amax_input_wrangler,
    ),
    TorchLibOpInfo(
        "any",
        core_ops.aten_any,
    ).skip(
        matcher=lambda sample: len(sample.kwargs) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo(
        "any_dim",
        core_ops.aten_any_dim,
    ).skip(
        matcher=lambda sample: not (len(sample.kwargs) > 0)
        or isinstance(sample.kwargs.get("dim"), tuple),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design. dim must be an integer",
    ),
    TorchLibOpInfo("any_dims", core_ops.aten_any_dims).skip(
        matcher=lambda sample: not isinstance(sample.kwargs.get("dim"), tuple),
        reason="this overload requires dim to be a tuple",
    ),
    TorchLibOpInfo("asin", core_ops.aten_asin),
    TorchLibOpInfo("asinh", core_ops.aten_asinh),
    TorchLibOpInfo("atan", core_ops.aten_atan),
    TorchLibOpInfo(
        "atan2", core_ops.aten_atan2, tolerance={torch.float16: (1e-3, 1e-3)}
    ),
    TorchLibOpInfo("atanh", core_ops.aten_atanh),
    TorchLibOpInfo("atleast_1d", core_ops.aten_atleast_1d).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "atleast_1d_Sequence",
        core_ops.aten_atleast_1d_sequence,
    )
    .skip(
        matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
        reason="takes tensor sequences only",
    )
    .xfail(
        reason=(
            "fixme: ORT shape inference failed."
            "https://github.com/microsoft/onnxscript/issues/1007"
        ),
    ),
    TorchLibOpInfo("atleast_2d", core_ops.aten_atleast_2d).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "atleast_2d_Sequence",
        core_ops.aten_atleast_2d_sequence,
    )
    .skip(
        matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
        reason="takes tensor sequences only",
    )
    .xfail(
        reason=(
            "fixme: ORT shape inference failed."
            "https://github.com/microsoft/onnxscript/issues/1007"
        ),
    ),
    TorchLibOpInfo("atleast_3d", core_ops.aten_atleast_3d).skip(
        matcher=lambda sample: isinstance(sample.input, (list, tuple)),
        reason="takes single tensor as input",
    ),
    TorchLibOpInfo(
        "atleast_3d_Sequence",
        core_ops.aten_atleast_3d_sequence,
    )
    .skip(
        matcher=lambda sample: not isinstance(sample.input, (list, tuple)),
        reason="takes tensor sequences only",
    )
    .xfail(
        reason=(
            "fixme: ORT shape inference failed."
            "https://github.com/microsoft/onnxscript/issues/1007"
        ),
    ),
    TorchLibOpInfo(
        "baddbmm", core_ops.aten_baddbmm, tolerance={torch.float16: (1e-3, 1e-2)}
    ),
    TorchLibOpInfo("bernoulli", core_ops.aten_bernoulli, nondeterministic=True),
    TorchLibOpInfo(
        # This string is a unique ID. In extra_opinfo.py, we
        # also define test data for this ID with
        # `opinfo_core.OpInfo("aten.bernoulli.p", ...)`.
        "ops.aten.bernoulli.p",
        core_ops.aten_bernoulli_p,
        # Skip comparison for the output of this op because it is a random tensor.
        nondeterministic=True,
    ),
    TorchLibOpInfo("ops.aten.bernoulli.p_deterministic", core_ops.aten_bernoulli_p),
    TorchLibOpInfo("bitwise_and", core_ops.aten_bitwise_and),
    TorchLibOpInfo("bitwise_left_shift_int16", core_ops.aten_bitwise_left_shift_int16),
    TorchLibOpInfo("bitwise_left_shift_int32", core_ops.aten_bitwise_left_shift_int32),
    TorchLibOpInfo("bitwise_left_shift_int64", core_ops.aten_bitwise_left_shift_int64),
    TorchLibOpInfo("bitwise_left_shift_int8", core_ops.aten_bitwise_left_shift_int8),
    TorchLibOpInfo("bitwise_not", core_ops.aten_bitwise_not),
    TorchLibOpInfo("bitwise_or", core_ops.aten_bitwise_or),
    TorchLibOpInfo(
        "bitwise_right_shift_int16", core_ops.aten_bitwise_right_shift_int16
    ),
    TorchLibOpInfo(
        "bitwise_right_shift_int32", core_ops.aten_bitwise_right_shift_int32
    ),
    TorchLibOpInfo(
        "bitwise_right_shift_int64", core_ops.aten_bitwise_right_shift_int64
    ),
    TorchLibOpInfo("bitwise_right_shift_int8", core_ops.aten_bitwise_right_shift_int8),
    TorchLibOpInfo("bitwise_xor", core_ops.aten_bitwise_xor),
    TorchLibOpInfo("ops.aten.blackman_window", core_ops.aten_blackman_window),
    TorchLibOpInfo("bmm", core_ops.aten_bmm),
    TorchLibOpInfo("broadcast_to", core_ops.aten_broadcast_to),
    TorchLibOpInfo("cat", core_ops.aten_cat).skip(
        matcher=lambda sample: sample.input[0].equal(
            torch.tensor([]).to(sample.input[0].device)
        ),
        reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
    ),
    TorchLibOpInfo("cat", core_ops.aten_cat_complex, complex=True).skip(
        matcher=lambda sample: sample.input[0].equal(
            torch.tensor([]).to(sample.input[0].device)
        ),
        reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
    ),
    TorchLibOpInfo("ceil", core_ops.aten_ceil),
    TorchLibOpInfo(
        "chunk",
        core_ops.aten_chunk,
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not implement SplitToSequence for bool inputs: https://github.com/microsoft/onnxruntime/issues/16905",
    ),
    TorchLibOpInfo("clamp_max", core_ops.aten_clamp_max).skip(
        reason="Size 0 inputs are not handled by design",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo("clamp_min", core_ops.aten_clamp_min).skip(
        reason="Size 0 inputs are not handled by design",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo("clone", core_ops.aten_clone),
    TorchLibOpInfo("complex", core_ops.aten_complex),
    TorchLibOpInfo("concat", core_ops.aten_cat).skip(
        matcher=lambda sample: sample.input[0].equal(
            torch.tensor([]).to(sample.input[0].device)
        ),
        reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
    ),
    TorchLibOpInfo("concatenate", core_ops.aten_cat).skip(
        matcher=lambda sample: sample.input[0].equal(
            torch.tensor([]).to(sample.input[0].device)
        ),
        reason="fixme: ORT aborts with zero-dim tensors. https://github.com/microsoft/onnxruntime/issues/16619",
    ),
    TorchLibOpInfo("conj", core_ops.aten_conj),
    TorchLibOpInfo("conj", core_ops.aten_conj_complex, complex=True),
    TorchLibOpInfo("constant_pad_nd", core_ops.aten_constant_pad_nd),
    # TorchLibOpInfo("copy", core_ops.aten_copy),  # copy is not in OPS_DB
    TorchLibOpInfo("cos", core_ops.aten_cos),
    TorchLibOpInfo("cosh", core_ops.aten_cosh),
    TorchLibOpInfo(
        "cross", core_ops.aten_cross, tolerance={torch.float16: (6e-3, 3e-3)}
    ),
    TorchLibOpInfo("deg2rad", core_ops.aten_deg2rad),
    # TorchLibOpInfo("detach", core_ops.aten_detach),  # detach is not in OP-TEST-DB
    TorchLibOpInfo("diagonal", core_ops.aten_diagonal),
    TorchLibOpInfo("diagonal_bool", core_ops.aten_diagonal_bool),
    TorchLibOpInfo("div", core_ops.aten_div).skip(
        matcher=lambda sample: sample.kwargs.get("rounding_mode") is not None,
        reason="this variation does not take the rounding_mode argument",
    ),
    TorchLibOpInfo("true_divide", core_ops.aten_div),
    TorchLibOpInfo("true_divide", core_ops.aten_div_complex, complex=True),
    TorchLibOpInfo("div_mode", core_ops.aten_div_mode)
    .skip(
        variant_name="no_rounding_mode",
        reason="this variation requires the rounding_mode argument",
    )
    .skip(
        variant_name="trunc_rounding",
        dtypes=(torch.float16,),
        # Numbers match sometimes but not other times
        reason="fixme: off-by-one. https://github.com/microsoft/onnxscript/issues/990",
    ),
    TorchLibOpInfo("div_mode_int", core_ops.aten_div_mode_int).skip(
        variant_name="no_rounding_mode",
        reason="this variation requires the rounding_mode argument",
    ),
    TorchLibOpInfo("dot", core_ops.aten_dot),
    TorchLibOpInfo(
        "empty",
        core_ops.aten_empty,
        input_wrangler=_empty_input_wrangler,
        nondeterministic=True,
    ),
    TorchLibOpInfo(
        "einsum", core_ops.aten_einsum, input_wrangler=_einsum_input_wrangler
    )
    .xfail(
        reason="fixme: PyTorch produces int64 output with int32 input",
        dtypes=(torch.int32,),
    )
    .xfail(
        reason="fixme: ONNX shape inference fails: https://github.com/onnx/onnx/issues/5739",
        matcher=lambda sample: sample.args[0] == "...ik, ...j -> ij",
    ),
    # TorchLibOpInfo("empty_strided", core_ops.aten_empty_strided),  # empty_strided is not in OPS_DB
    TorchLibOpInfo("eq", core_ops.aten_eq),
    TorchLibOpInfo("equal", core_ops.aten_equal),
    TorchLibOpInfo("exp", core_ops.aten_exp),
    TorchLibOpInfo("exp2", core_ops.aten_exp2),
    TorchLibOpInfo("expand", core_ops.aten_expand),
    TorchLibOpInfo("expand_as", core_ops.aten_expand_as),
    TorchLibOpInfo("erf", special_ops.aten_special_erf),
    TorchLibOpInfo(
        "erfc", special_ops.aten_special_erfc, tolerance={torch.float16: (5e-1, 2e-4)}
    ),
    TorchLibOpInfo(
        "expm1", special_ops.aten_special_expm1, tolerance={torch.float16: (1e-2, 2e-4)}
    ),
    TorchLibOpInfo("special.erfcx", special_ops.aten_special_erfcx).xfail(
        reason="fixme: The implementation is numerically unstable: https://github.com/microsoft/onnxscript/issues/1223"
    ),
    TorchLibOpInfo("fill", core_ops.aten_fill),
    TorchLibOpInfo("flip", core_ops.aten_flip).skip(
        reason="fixme: size 0 inputs are not handled yet",
        matcher=lambda sample: sample.input.numel() == 0,
    ),
    TorchLibOpInfo("flatten", core_ops.aten_flatten),
    TorchLibOpInfo("floor", core_ops.aten_floor),
    TorchLibOpInfo("ops.aten.floor_divide", core_ops.aten_floor_divide),
    TorchLibOpInfo("fmod", core_ops.aten_fmod),
    TorchLibOpInfo("frac", core_ops.aten_frac),
    TorchLibOpInfo("full", core_ops.aten_full),
    TorchLibOpInfo(
        "full_like",
        core_ops.aten_full_like,
    ),
    TorchLibOpInfo("gather", core_ops.aten_gather),
    TorchLibOpInfo("ge", core_ops.aten_ge),
    TorchLibOpInfo("ge_bool", core_ops.aten_ge_bool),
    TorchLibOpInfo("gt", core_ops.aten_gt),
    TorchLibOpInfo("gt_bool", core_ops.aten_gt_bool),
    # TorchLibOpInfo("is_same_size", core_ops.aten_is_same_size),  # no test case in OPS_DB
    # TorchLibOpInfo("is_nonzero", core_ops.aten_is_nonzero),  # no test case in OPS_DB
    TorchLibOpInfo("ops.aten.index.Tensor", core_ops.aten_index),
    TorchLibOpInfo("ops.aten.index.Tensor.bool", core_ops.aten_index_bool),
    TorchLibOpInfo(
        "index_put_bool",
        core_ops.aten_index_put_bool,
    ).skip(
        matcher=lambda sample: sample.args[0][0].dtype != torch.bool,
        reason="this Aten overload only supports tensor(bool) as indices",
    ),
    TorchLibOpInfo(
        "index_put",
        core_ops.aten_index_put,
    )
    .skip(
        matcher=lambda sample: sample.args[0][0].dtype != torch.int64,
        reason="this Aten overload only supports tensor(int) as indices",
    )
    .xfail(
        dtypes=(torch.float16,),
        matcher=lambda sample: sample.kwargs.get("accumulate") is True,
        reason="fixme: ORT only supports float32 when accumulate is True:  MLFloat16 data type is not supported with ScatterND when reduction is 'add'",
    ),
    TorchLibOpInfo("ops.aten.index_put", core_ops.aten_index_put),
    TorchLibOpInfo("ops.aten._unsafe_index_put", core_ops.aten_index_put),
    TorchLibOpInfo("index_select", core_ops.aten_index_select),
    TorchLibOpInfo("isclose", core_ops.aten_isclose),
    TorchLibOpInfo("isfinite", core_ops.aten_isfinite),
    TorchLibOpInfo("isinf", core_ops.aten_isinf),
    TorchLibOpInfo("isnan", core_ops.aten_isnan),
    TorchLibOpInfo("isneginf", core_ops.aten_isneginf),
    TorchLibOpInfo("isposinf", core_ops.aten_isposinf),
    TorchLibOpInfo("lift_fresh_copy", core_ops.aten_lift_fresh_copy),
    TorchLibOpInfo("linalg.det", linalg_ops.aten_linalg_det),
    TorchLibOpInfo(
        "linalg.vector_norm",
        linalg_ops.aten_linalg_vector_norm,
        tolerance={torch.float16: (2e-3, 2e-3)},
        input_wrangler=_linalg_vector_norm_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("ord") == 6,
        dtypes=(torch.float16,),
        reason="ORT returns a more accurate value for float16 with ord=6 (expected=Inf, actual=9.48).",
    ),
    TorchLibOpInfo(
        "linspace",
        core_ops.aten_linspace,
        tolerance={torch.float16: (2e-2, 2e-3)},
    ).xfail(
        dtypes=(torch.int64, torch.int32),
        reason="fixme: Results do not match with PyTorch. https://github.com/microsoft/onnxscript/issues/854",
    ),
    TorchLibOpInfo("log", core_ops.aten_log),
    TorchLibOpInfo("le", core_ops.aten_le),
    TorchLibOpInfo("le_bool", core_ops.aten_le_bool),
    TorchLibOpInfo(
        "lerp",
        core_ops.aten_lerp,
        tolerance={torch.float16: (2e-3, 2e-1)},
    ),
    TorchLibOpInfo("log10", core_ops.aten_log10),
    TorchLibOpInfo("log1p", core_ops.aten_log1p),
    TorchLibOpInfo(
        "log_softmax",
        special_ops.aten_special_log_softmax,
        tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (4e-4, 6e-3)},
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    )
    .xfail(
        variant_name="with_dtype",
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    )
    .skip(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    )
    .skip(
        variant_name="with_dtype",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    ),
    TorchLibOpInfo("log2", core_ops.aten_log2),
    TorchLibOpInfo(
        "logaddexp", core_ops.aten_logaddexp, tolerance={torch.float16: (1, 1e-4)}
    ),
    TorchLibOpInfo(
        "logaddexp2", core_ops.aten_logaddexp2, tolerance={torch.float16: (2e-2, 6e-4)}
    ),
    TorchLibOpInfo(
        "logcumsumexp",
        core_ops.aten_logcumsumexp,
        tolerance={torch.float16: (1e-2, 1e-1)},
    ),
    TorchLibOpInfo("logdet", core_ops.aten_logdet),
    TorchLibOpInfo("logsumexp", core_ops.aten_logsumexp),
    TorchLibOpInfo("lt", core_ops.aten_lt),
    TorchLibOpInfo("lt_bool", core_ops.aten_lt_bool),
    TorchLibOpInfo("masked_fill", core_ops.aten_masked_fill).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not have an implementation for Where with bool inputs.",
    ),
    TorchLibOpInfo(
        "matmul",
        core_ops.aten_matmul,
        # Windows requires a more relaxed tolerance
        tolerance={torch.float32: (2e-5, 2e-5), torch.float16: (2e-3, 2e-2)},
    ).skip(
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
    ),
    TorchLibOpInfo("maximum", core_ops.aten_maximum),
    TorchLibOpInfo("maximum_bool", core_ops.aten_maximum_bool),
    TorchLibOpInfo(
        "mean",
        core_ops.aten_mean,
        input_wrangler=_mean_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is not None,
        reason="this Aten overload only accept 1 inputs: self",
    ),
    TorchLibOpInfo(
        "mean_dim",
        core_ops.aten_mean_dim,
        input_wrangler=_mean_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is None,
        reason="this Aten overload can accept 2 inputs:(self, dim)",
    ),
    TorchLibOpInfo("mH", core_ops.aten_mH),
    TorchLibOpInfo("mH", core_ops.aten_mH_complex, complex=True),
    TorchLibOpInfo("min_dim", core_ops.aten_min_dim).xfail(
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and not isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another int as args",
    ),
    TorchLibOpInfo(
        "min",
        core_ops.aten_min,
    ).skip(
        matcher=lambda sample: len(sample.args) > 0,
        reason="this ATen overload only supports one tensor as input by design",
    ),
    TorchLibOpInfo("minimum", core_ops.aten_minimum),
    TorchLibOpInfo("minimum_bool", core_ops.aten_minimum_bool),
    TorchLibOpInfo("mm", core_ops.aten_mm).skip(
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
    ),
    TorchLibOpInfo("mT", core_ops.aten_mT),
    TorchLibOpInfo("mT", core_ops.aten_mT_complex, complex=True),
    TorchLibOpInfo("mul", core_ops.aten_mul),
    TorchLibOpInfo("mul", core_ops.aten_mul_complex, complex=True),
    TorchLibOpInfo(
        "mv",
        core_ops.aten_mv,
        tolerance={torch.float16: (3e-2, 1e-2)},
    ),
    TorchLibOpInfo("narrow", core_ops.aten_narrow),
    TorchLibOpInfo("ops.aten.native_dropout", core_ops.aten_native_dropout),
    TorchLibOpInfo("ne", core_ops.aten_ne),
    TorchLibOpInfo("neg", core_ops.aten_neg),
    TorchLibOpInfo(
        "new_empty",
        core_ops.aten_new_empty,
        nondeterministic=True,
    ),
    TorchLibOpInfo(
        "new_empty_strided",
        core_ops.aten_new_empty_strided,
        nondeterministic=True,
    ),
    TorchLibOpInfo(
        "new_full",
        core_ops.aten_new_full,
    ),
    TorchLibOpInfo(
        "new_ones",
        core_ops.aten_new_ones,
    ),
    TorchLibOpInfo(
        "new_zeros",
        core_ops.aten_new_zeros,
    ),
    TorchLibOpInfo("nn.functional.celu", nn_ops.aten_celu),
    TorchLibOpInfo("nn.functional.celu_type_promoted", nn_ops.aten_celu_type_promoted),
    TorchLibOpInfo(
        "nn.functional.cross_entropy",
        # use cross_entropy as test case instead of cross_entropy_loss (not in OPS_DB)
        nn_ops.aten_cross_entropy_loss,
        tolerance={torch.float16: (1e-2, 1e-2)},
        input_wrangler=_cross_entropy_input_wrangler,
    ).xfail(
        matcher=lambda sample: len(sample.args) < 1
        or (
            isinstance(sample.args[0], torch.Tensor)
            and sample.args[0].dtype != torch.int64
        ),
        reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[target] as int type",
    ),
    TorchLibOpInfo(
        "nn.functional.dropout",
        core_ops.aten_dropout,
        input_wrangler=_dropout_input_wrangler,
    ).skip(
        matcher=lambda sample: len(sample.kwargs) == 0
        or sample.kwargs.get("p", 0.0) > 0.0,
        reason="dropout is random so the result not match",
    ),
    TorchLibOpInfo("nn.functional.elu", nn_ops.aten_elu),
    TorchLibOpInfo(
        "ops.aten.embedding_bag",
        core_ops.aten_embedding_bag,
        tolerance={torch.float16: (1e-2, 5e-2)},
        compare_shape_only_for_output=(1, 2, 3),
    ),
    TorchLibOpInfo(
        "ops.aten.embedding_bag.padding_idx",
        core_ops.aten_embedding_bag_padding_idx,
        tolerance={torch.float16: (1e-2, 1e-2)},
        compare_shape_only_for_output=(1, 2, 3),
    ),
    TorchLibOpInfo(
        "ops.aten.embedding_renorm",
        core_ops.aten_embedding_renorm,
        tolerance={torch.float16: (1e-2, 1e-2)},
        compare_shape_only_for_output=(1, 2, 3),
    ),
    TorchLibOpInfo(
        "nn.functional.embedding",
        core_ops.aten_embedding,
        input_wrangler=_embedding_input_wrangler,
    ),
    TorchLibOpInfo("nn.functional.hardsigmoid", nn_ops.aten_hardsigmoid),
    TorchLibOpInfo("nn.functional.hardswish", nn_ops.aten_hardswish),
    TorchLibOpInfo("nn.functional.hardtanh", nn_ops.aten_hardtanh),
    TorchLibOpInfo("nn.functional.leaky_relu", nn_ops.aten_leaky_relu),
    TorchLibOpInfo(
        "nn.functional.logsigmoid",
        nn_ops.aten_log_sigmoid,
        tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (8e-2, 4e-4)},
    ),
    TorchLibOpInfo("nn.functional.mish", nn_ops.aten_mish),
    TorchLibOpInfo(
        "nn.functional.nll_loss",
        nn_ops.aten_nll_loss,
        input_wrangler=_nll_loss_input_wrangler,
        tolerance={torch.float16: (5e-2, 1e-2)},
    ),
    TorchLibOpInfo(
        "nn.functional.pixel_shuffle",
        core_ops.aten_pixel_shuffle,
    )
    .xfail(
        dtypes=(torch.int32, torch.int64),
        reason="fixme: ONNX Runtime does not support int32/64 inputs",
    )
    .xfail(
        matcher=lambda sample: sample.input.numel() == 0,
        reason="fixme: ORT does not support empty tensor as input",
    ),
    TorchLibOpInfo(
        "nn.functional.pixel_unshuffle",
        core_ops.aten_pixel_unshuffle,
    )
    .xfail(
        dtypes=(torch.int32, torch.int64),
        reason="fixme: ONNX Runtime does not support int32/64 inputs",
    )
    .xfail(
        matcher=lambda sample: sample.input.numel() == 0,
        reason="fixme: ORT does not support empty tensor as input",
    ),
    TorchLibOpInfo(
        "ops.aten.reflection_pad1d",
        nn_ops.aten_reflection_pad1d,
    ).xfail(
        dtypes=(torch.int64,),
        reason="Torch not implement reflection_pad1d for int64.",
    ),
    TorchLibOpInfo(
        "nn.functional.reflection_pad2d",
        nn_ops.aten_reflection_pad2d,
        input_wrangler=_reflection_pad2d_input_wrangler,
    ).skip(
        matcher=lambda sample: not (
            len(sample.args) > 1 and sample.args[1] == "reflect"
        ),
        reason="this Aten overload need args[1] == 'reflect' for pad mode",
    ),
    TorchLibOpInfo("nn.functional.relu", nn_ops.aten_relu),
    TorchLibOpInfo("nn.functional.relu6", nn_ops.aten_relu6),
    TorchLibOpInfo(
        "ops.aten.replication_pad1d",
        nn_ops.aten_replication_pad1d,
    ),
    TorchLibOpInfo(
        "nn.functional.replication_pad2d",
        nn_ops.aten_replication_pad2d,
        input_wrangler=_replication_pad2d_input_wrangler,
    ).skip(
        matcher=lambda sample: not (
            len(sample.args) > 1 and sample.args[1] == "replicate"
        ),
        reason="this Aten overload need args[1] == 'replicate' for pad mode",
    ),
    TorchLibOpInfo(
        "nn.functional.replication_pad3d",
        nn_ops.aten_replication_pad3d,
        input_wrangler=_replication_pad3d_input_wrangler,
    ).skip(
        matcher=lambda sample: not (
            len(sample.args) > 1
            and sample.args[1] == "replicate"
            and len(sample.input.shape) == 5
        ),
        reason="this Aten overload need args[1] == 'replicate' for pad mode, and 3D tensor",
    ),
    TorchLibOpInfo("nn.functional.selu", core_ops.aten_selu),
    TorchLibOpInfo(
        "nn.functional.mse_loss",
        nn_ops.aten_mse_loss,
        input_wrangler=_mse_loss_input_wrangler,
    ),
    TorchLibOpInfo(
        "nonzero",
        core_ops.aten_nonzero,
        input_wrangler=_nonzero_input_wrangler,
    )
    .xfail(
        matcher=lambda sample: sample.kwargs.get("as_tuple"),
        reason="as_tuple=True is not supported",
    )
    .xfail(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0]).",
    ),
    TorchLibOpInfo("normal", core_ops.aten_normal, nondeterministic=True)
    .skip(
        matcher=lambda sample: len(sample.args) > 0
        and not isinstance(sample.args[0], float),
        reason="ORT only accept float type for args[0] 'mean'",
    )
    .xfail(
        variant_name="number_mean",
        reason="This variant does not support dtype as an argument",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
    ),
    TorchLibOpInfo(
        "ops.aten.normal.float_Tensor",
        core_ops.aten_normal_float_tensor,
        nondeterministic=True,
    ),
    TorchLibOpInfo(
        "ops.aten.normal.Tensor_float",
        core_ops.aten_normal_tensor_float,
        nondeterministic=True,
    ),
    TorchLibOpInfo(
        "ops.aten.normal.Tensor_Tensor",
        core_ops.aten_normal_tensor_tensor,
        nondeterministic=True,
    ),
    TorchLibOpInfo("ones", core_ops.aten_ones),
    TorchLibOpInfo(
        "permute",
        core_ops.aten_permute,
        input_wrangler=_permute_input_wrangler,
    ),
    TorchLibOpInfo("polar", core_ops.aten_polar),
    TorchLibOpInfo("pow", core_ops.aten_pow),
    TorchLibOpInfo("prod", core_ops.aten_prod).skip(
        matcher=lambda sample: sample.kwargs.get("dim") is not None
        or sample.kwargs.get("keepdim") is not None
        or sample.kwargs.get("dtype") != -1,
        reason="this Aten overload only accept 1 inputs: self",
    ),
    TorchLibOpInfo("prod_dim_int", core_ops.aten_prod_dim_int).skip(
        matcher=lambda sample: (
            sample.kwargs.get("dim") is None and sample.kwargs.get("keepdim") is None
        )
        or sample.kwargs.get("dtype") != -1,
        reason="this Aten overload can accept 3 inputs:(self, dim, keepdim)",
    ),
    TorchLibOpInfo("nn.functional.prelu", core_ops.aten_prelu),
    TorchLibOpInfo("ops.aten.rand", core_ops.aten_rand, nondeterministic=True),
    TorchLibOpInfo(
        "ops.aten.rand_like", core_ops.aten_rand_like, nondeterministic=True
    ),
    TorchLibOpInfo("ops.aten.randint", core_ops.aten_randint, nondeterministic=True),
    TorchLibOpInfo(
        "ops.aten.randint.low", core_ops.aten_randint_low, nondeterministic=True
    ),
    TorchLibOpInfo(
        "ops.aten.randint_like", core_ops.aten_randint_like, nondeterministic=True
    ),
    TorchLibOpInfo(
        "ops.aten.randint_like.low_dtype",
        core_ops.aten_randint_like_low_dtype,
        nondeterministic=True,
    ),
    TorchLibOpInfo("ops.aten.randn", core_ops.aten_randn, nondeterministic=True).xfail(
        dtypes=(torch.float16,),
        reason="fixme: Shape inference error",
    ),
    TorchLibOpInfo(
        "ops.aten.randn_like", core_ops.aten_randn_like, nondeterministic=True
    ),
    TorchLibOpInfo("rad2deg", core_ops.aten_rad2deg),
    TorchLibOpInfo("reciprocal", core_ops.aten_reciprocal),
    TorchLibOpInfo(
        "remainder",
        core_ops.aten_remainder,
    ),
    TorchLibOpInfo("repeat", core_ops.aten_repeat),
    TorchLibOpInfo("reshape", core_ops.aten_reshape),
    TorchLibOpInfo("resolve_conj", core_ops.aten_resolve_conj),
    TorchLibOpInfo("resolve_neg", core_ops.aten_resolve_neg),
    TorchLibOpInfo("round", core_ops.aten_round)
    .xfail(
        variant_name="decimals_3",
        reason="This variant does not accept decimals",
    )
    .xfail(
        variant_name="decimals_neg_3",
        reason="This variant does not accept decimals",
    ),
    TorchLibOpInfo("round_decimals", core_ops.aten_round_decimals),
    TorchLibOpInfo("rsqrt", core_ops.aten_rsqrt),
    TorchLibOpInfo(
        "scalar_tensor",
        core_ops.aten_scalar_tensor,
        input_wrangler=_scalar_tensor_input_wrangler,
    ),
    TorchLibOpInfo(
        "scalar_tensor",
        core_ops.aten_scalar_tensor,
        input_wrangler=_scalar_tensor_input_wrangler,
        complex=True,
    ),
    TorchLibOpInfo(
        "ops.aten.scalar_tensor",
        core_ops.aten_scalar_tensor_complex,
        complex=True,
    ),
    TorchLibOpInfo(
        "scatter_add",
        core_ops.aten_scatter_add,
    )
    .xfail(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch. https://github.com/onnx/onnx/issues/4986",
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT error: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'",
    ),
    TorchLibOpInfo("select", core_ops.aten_select),
    TorchLibOpInfo("select_scatter", core_ops.aten_select_scatter),
    TorchLibOpInfo("sigmoid", core_ops.aten_sigmoid),
    TorchLibOpInfo("sign", core_ops.aten_sign),
    TorchLibOpInfo("nn.functional.silu", nn_ops.aten_silu),
    TorchLibOpInfo("sin", core_ops.aten_sin),
    TorchLibOpInfo(
        "sinc", special_ops.aten_special_sinc, tolerance={torch.float16: (1e-2, 6e-4)}
    ),
    TorchLibOpInfo("sinh", core_ops.aten_sinh),
    TorchLibOpInfo(
        "softmax",
        core_ops.aten_softmax,
        tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (3e-4, 4e-4)},
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    )
    .xfail(
        variant_name="with_dtype",
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    )
    .skip(
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: SoftMax does not support empty tensor as input",
    )
    .skip(
        variant_name="with_dtype",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: SoftMax does not support empty tensor as input",
    ),
    TorchLibOpInfo("nn.functional.softplus", nn_ops.aten_softplus),
    TorchLibOpInfo("sort", core_ops.aten_sort).xfail(
        dtypes=(torch.float16,),
        reason="fixme: Tensor-likes are not close. Tests pass for float32.",
    ),
    TorchLibOpInfo(
        "split_with_sizes",
        core_ops.aten_split_with_sizes,
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not implement SplitToSequence for bool inputs: https://github.com/microsoft/onnxruntime/issues/16905",
    ),
    TorchLibOpInfo(
        "split",
        core_ops.aten_split,
    )
    .xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not implement SplitToSequence for bool inputs: https://github.com/microsoft/onnxruntime/issues/16905",
    )
    .xfail(
        variant_name="list_args",
        dtypes=(torch.bool,),
        reason="fixme: ORT does not implement SplitToSequence for bool inputs: https://github.com/microsoft/onnxruntime/issues/16905",
    ),
    TorchLibOpInfo("sqrt", core_ops.aten_sqrt),
    TorchLibOpInfo(
        "squeeze_dim",
        core_ops.aten_squeeze_dim,
    )
    .skip(
        matcher=lambda sample: not (
            len(sample.args) > 0 and isinstance(sample.args[0], int)
        ),
        reason="this Aten overload only support one tensor as input and one int as args by design",
    )
    .skip(
        matcher=lambda sample: len(sample.input.shape) != 0
        and sample.input.shape[sample.args[0]] != 1,
        reason="this Aten overload only support squeeze dim with size 1",
    ),
    TorchLibOpInfo(
        "squeeze_dim",
        core_ops.aten_squeeze_dim_complex,
        complex=True,
    )
    .skip(
        matcher=lambda sample: not (
            len(sample.args) > 0 and isinstance(sample.args[0], int)
        ),
        reason="this Aten overload only support one tensor as input and one int as args by design",
    )
    .skip(
        matcher=lambda sample: len(sample.input.shape) != 0
        and sample.input.shape[sample.args[0]] != 1,
        reason="this Aten overload only support squeeze dim with size 1",
    ),
    TorchLibOpInfo(
        "squeeze",
        core_ops.aten_squeeze,
    ).skip(
        matcher=lambda sample: len(sample.args) != 0,
        reason="this Aten overload only support one tensor as input by design",
    ),
    TorchLibOpInfo("stack", core_ops.aten_stack),
    TorchLibOpInfo("stack", core_ops.aten_stack_complex, complex=True),
    TorchLibOpInfo("sub", core_ops.aten_sub, tolerance={torch.float16: (2e-3, 1e-3)}),
    TorchLibOpInfo("sub", core_ops.aten_sub_complex, complex=True),
    # TorchLibOpInfo("sym_size", core_ops.aten_sym_size),  # no test case in OPS_DB
    TorchLibOpInfo("t", core_ops.aten_t),
    TorchLibOpInfo("tan", core_ops.aten_tan),
    TorchLibOpInfo("tanh", core_ops.aten_tanh),
    TorchLibOpInfo(
        "tile",
        core_ops.aten_tile,
    ).skip(
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape)
        or not sample.input.shape,
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
    TorchLibOpInfo("topk", core_ops.aten_topk)
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
    TorchLibOpInfo("tril", core_ops.aten_tril).xfail(
        dtypes=(torch.int32,),
        reason="fixme: ORT does not have an implementation of Trilu for int32.",
    ),
    TorchLibOpInfo("triu", core_ops.aten_triu).xfail(
        dtypes=(torch.int32,),
        reason="fixme: ORT does not have an implementation of Trilu for int32.",
    ),
    TorchLibOpInfo("trunc", core_ops.aten_trunc),
    TorchLibOpInfo(
        "unbind",
        core_ops.aten_unbind,
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not implement SplitToSequence for bool inputs: https://github.com/microsoft/onnxruntime/issues/16905",
    ),
    TorchLibOpInfo(
        "unflatten",
        core_ops.aten_unflatten,
        input_wrangler=_unflatten_input_wrangler,
    ).xfail(
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
    TorchLibOpInfo("unfold", core_ops.aten_unfold),
    TorchLibOpInfo("ops.aten.unfold", core_ops.aten_unfold),
    TorchLibOpInfo("unsqueeze", core_ops.aten_unsqueeze),
    TorchLibOpInfo("view", core_ops.aten_view),
    TorchLibOpInfo("view", core_ops.aten_view_complex, complex=True),
    TorchLibOpInfo("view_as", core_ops.aten_view_as),
    TorchLibOpInfo("view_as_complex", core_ops.aten_view_as_complex),
    TorchLibOpInfo("view_as_complex_copy", core_ops.aten_view_as_complex_copy),
    TorchLibOpInfo("view_as_real", core_ops.aten_view_as_real, complex=True),
    TorchLibOpInfo("view_as_real_copy", core_ops.aten_view_as_real_copy, complex=True),
    TorchLibOpInfo("view_copy", core_ops.aten_view_copy),
    TorchLibOpInfo(
        "where", core_ops.aten_where, input_wrangler=_where_input_wrangler
    ).xfail(
        dtypes=(torch.bool,),
        reason="fixme: ORT does not have an implementation for Where with bool inputs.",
    ),
    TorchLibOpInfo("xlogy", special_ops.aten_special_xlogy),
    TorchLibOpInfo("zeros", core_ops.aten_zeros),
    TorchLibOpInfo(
        "arange_start_step",
        core_ops.aten_arange_start_step,
    )
    .skip(
        matcher=lambda sample: len(sample.args) != 2,
        reason="arange_start_step overload takes three arguments (input, start, step)",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="dtype needs to be specified for non-float tensors",
        dtypes=(torch.float16, torch.int64, torch.int32),
    ),
    TorchLibOpInfo(
        "arange_start",
        core_ops.aten_arange_start,
    )
    .skip(
        matcher=lambda sample: len(sample.args) != 1,
        reason="arange_start overload takes two arguments (input, start)",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="dtype needs to be specified for non-float tensors",
        dtypes=(torch.float16, torch.int64, torch.int32),
    ),
    TorchLibOpInfo(
        "arange",
        core_ops.aten_arange,
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
    ),
    TorchLibOpInfo("argmax", core_ops.aten_argmax),
    TorchLibOpInfo("argmin", core_ops.aten_argmin),
    TorchLibOpInfo(
        "as_strided",
        core_ops.aten_as_strided,
    ).xfail(
        variant_name="partial_views",
        reason="ONNX doesn't have partial view for tensor",
    ),
    TorchLibOpInfo("clamp", core_ops.aten_clamp),
    TorchLibOpInfo(
        "ops.aten.col2im",
        nn_ops.aten_col2im,
    ).xfail(
        dtypes=(torch.float16,),
        reason="fixme: Tensor-likes are not close. https://github.com/microsoft/onnxruntime/issues/16007",
    ),
    TorchLibOpInfo("cumsum", core_ops.aten_cumsum).xfail(
        dtypes=(torch.int32,),
        reason="fixme: torch.cumsum with int32 inputs uses int64 as the output type",
    ),
    TorchLibOpInfo("contiguous", core_ops.aten_contiguous),
    TorchLibOpInfo(
        "ops.aten.convolution",
        core_ops.aten_convolution,
        tolerance={torch.float32: (3.7e-5, 1.8e-4)},
    ),
    TorchLibOpInfo("empty_like", core_ops.aten_empty_like, nondeterministic=True),
    TorchLibOpInfo(
        "grid_sampler_2d",
        core_ops.aten_grid_sampler_2d,
    )
    .skip(
        # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
        matcher=lambda sample: sample.args[1] == 2,
        reason="fixme: 'bicubic' mode in ORT implemented differently with Torch",
    )
    .skip(
        dtypes=(torch.float16,),
        reason="fixme: Accuracy is not high enough",
    ),
    TorchLibOpInfo(
        "nn.functional.group_norm",
        nn_ops.aten_group_norm,
        tolerance={torch.float16: (1e-2, 7e-3)},
    ).xfail(
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
        reason="Using op.InstanceNormalization to simulate GroupNorm, which does not support 0-dim input",
    ),
    TorchLibOpInfo(
        "ops.aten.hamming_window",
        core_ops.aten_hamming_window,
        tolerance={torch.float32: (8e-2, 6e-3)},
    ),
    TorchLibOpInfo("ops.aten.hann_window", core_ops.aten_hann_window),
    TorchLibOpInfo("heaviside", core_ops.aten_heaviside),
    TorchLibOpInfo(
        "nn.functional.grid_sample",
        core_ops.aten_grid_sampler,
        input_wrangler=_grid_sample_input_wrangler,
        tolerance={torch.float16: (8e-2, 2e-3)},
    ).skip(
        # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
        matcher=lambda sample: sample.kwargs.get("mode") == "bicubic"
        or len(sample.args[0].shape) != 4,
        reason="fixme: 'bicubic' mode in ORT implemented differently with Torch and only support 4D-tensor",
    ),
    TorchLibOpInfo(
        "ops.aten.layer_norm",
        core_ops.aten_layer_norm,
        tolerance={torch.float32: (3.7e-5, 1.8e-4)},
    ).xfail(
        dtypes=(torch.int64,),
        reason="fixme: ORT `LayerNormKernelImpl` not implemented for int64",
    ),
    TorchLibOpInfo(
        "logit", core_ops.aten_logit, tolerance={torch.float16: (1e-1, 7e-4)}
    ),
    TorchLibOpInfo("max_dim", core_ops.aten_max_dim).xfail(
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and not isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another int as args",
    ),
    TorchLibOpInfo(
        "max",
        core_ops.aten_max,
    ).skip(
        matcher=lambda sample: len(sample.args) > 0,
        reason="this ATen overload only supports one tensor as input by design",
    ),
    TorchLibOpInfo("multinomial", core_ops.aten_multinomial, nondeterministic=True),
    TorchLibOpInfo(
        # Custom from extra_opinfo
        "ops.aten.max_pool1d",
        nn_ops.aten_max_pool1d,
    ),
    TorchLibOpInfo(
        # Custom from extra_opinfo
        "ops.aten.max_pool2d",
        nn_ops.aten_max_pool2d,
    ),
    TorchLibOpInfo(
        "ops.aten.max_pool3d",  # Custom from extra_opinfo
        nn_ops.aten_max_pool3d,
    ).xfail(
        variant_name="empty_strides",
        reason="fixme: 'shape' do not match: torch.Size([2, 3, 4, 3]) != torch.Size([2, 3, 4, 2]). https://github.com/microsoft/onnxscript/issues/975",
    ),
    TorchLibOpInfo(
        "native_batch_norm",
        core_ops.aten_native_batch_norm,
        tolerance={torch.float16: (1e-2, 7e-3)},
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
    .skip(
        matcher=lambda sample: sample.kwargs.get("training") is True
        or sample.args[-3] is True,
        reason="fixme: ORT only supports BatchNorm less than opset14",
    ),
    TorchLibOpInfo(
        "ops.aten._native_batch_norm_legit",
        core_ops.aten_native_batch_norm,
        tolerance={torch.float16: (1e-2, 7e-3)},
    )
    .skip(
        device_type="cpu",
        matcher=lambda sample: sample.kwargs.get("training") is False,
        reason="native_batch_norm outputs different shapes on CPU and CUDA when training is False. Our implematation is based on that for CUDA",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("training") is True
        or sample.args[-3] is True,
        reason="fixme: ORT only supports BatchNorm less than opset14",
    ),
    TorchLibOpInfo(
        "ops.aten._native_batch_norm_legit.no_stats",
        core_ops.aten__native_batch_norm_no_stats,
    ),
    TorchLibOpInfo(
        "ops.aten._native_batch_norm_legit_functional",
        core_ops.aten__native_batch_norm_legit_functional,
        tolerance={torch.float16: (1e-2, 7e-3)},
    )
    .skip(
        device_type="cpu",
        matcher=lambda sample: sample.kwargs.get("training") is False,
        reason="native_batch_norm outputs different results on CPU and CUDA when training is False. Our implematation is based on that for CUDA",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("training") is True
        or sample.args[-3] is True,
        reason="fixme: ORT only supports BatchNorm less than opset14",
    ),
    TorchLibOpInfo(
        "ops.aten.native_group_norm",
        core_ops.aten_native_group_norm,
        tolerance={torch.float16: (1e-2, 7e-3)},
    ),
    TorchLibOpInfo(
        "native_layer_norm",
        core_ops.aten_native_layer_norm,
        tolerance={torch.float32: (3.7e-5, 1.8e-4), torch.float16: (1e-1, 7e-4)},
    )
    .xfail(
        dtypes=(torch.float32,),
        matcher=lambda sample: len(sample.input.shape) == 1,
        enabled_if=ops_test_common.IS_MACOS,
        reason="fixme: result mismatch. https://github.com/microsoft/onnxruntime/issues/20676",
    )
    .skip(
        dtypes=(torch.float16,),
        device_type="cpu",
        reason="native_layer_norm outputs different dtypes on CPU and CUDA. Our implematation is based on that for CUDA",
    ),
    TorchLibOpInfo(
        "nn.functional.avg_pool1d",
        nn_ops.aten_avg_pool1d,
        input_wrangler=_avg_pool_input_wrangler,
    )
    .xfail(
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)
        or (sample.kwargs.get("divisor_override") is not None),
        reason="ONNX doesn't support divisor_override argument",
    )
    .xfail(
        matcher=lambda sample: (sample.kwargs.get("ceil_mode") is True)
        and (
            sample.kwargs.get("count_include_pad") is True
            or sample.input.shape[2]
            % (
                sample.args[0][0]
                if isinstance(sample.args[0], tuple)
                else sample.args[0]
            )
            != 0
        ),
        reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19",
    ),
    TorchLibOpInfo(
        "nn.functional.avg_pool2d",
        nn_ops.aten_avg_pool2d,
        input_wrangler=_avg_pool_input_wrangler,
    ).xfail(
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)
        or (sample.kwargs.get("divisor_override") is not None),
        reason="ONNX doesn't support divisor_override argument",
    ),
    TorchLibOpInfo(
        "nn.functional.avg_pool3d",
        nn_ops.aten_avg_pool3d,
        input_wrangler=_avg_pool_input_wrangler,
    )
    .xfail(
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)
        or (sample.kwargs.get("divisor_override") is not None),
        reason="ONNX doesn't support divisor_override argument",
    )
    .xfail(
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True,
        reason="fixme(after opset19): ORT doesn't match PyTorch when ceil_mode=True until opset 19",
    ),
    TorchLibOpInfo(
        "nn.functional.conv1d",
        core_ops.aten_conv1d,
    ).xfail(
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv1d",
    ),
    TorchLibOpInfo(
        "nn.functional.conv2d",
        core_ops.aten_conv2d,
        tolerance={torch.float32: (2e-5, 3e-5)},
    ).xfail(
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv2d",
    ),
    TorchLibOpInfo(
        "nn.functional.instance_norm",
        core_ops.aten_instance_norm,
        tolerance={torch.float16: (1e-2, 1e-3)},
    ),
    TorchLibOpInfo(
        "ops.aten.conv3d",
        core_ops.aten_conv3d,
        tolerance={torch.float32: (3.7e-5, 1.8e-4)},
    ),
    TorchLibOpInfo(
        "nn.functional.gelu",
        nn_ops.aten_gelu,
        tolerance={torch.float16: (8e-2, 1e-4)},
    ),
    TorchLibOpInfo("nn.functional.glu", nn_ops.aten_glu),
    TorchLibOpInfo(
        "nn.functional.linear",
        nn_ops.aten_linear,
        tolerance={torch.float16: (1e-2, 1e-3)},
    ),
    TorchLibOpInfo(
        "nn.functional.unfold",
        nn_ops.aten_im2col,
        input_wrangler=_im2col_input_wrangler,
    ).xfail(
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape)
        or not sample.input.shape,
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool1d",
        nn_ops.aten_max_pool1d,
        input_wrangler=_max_pool_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is True,
        reason="this aten overload assume return_indices=False",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool1d_with_indices",
        nn_ops.aten_max_pool1d_with_indices,
        input_wrangler=_max_pool_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is False,
        reason="this aten overload assume return_indices=True",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool2d",
        nn_ops.aten_max_pool2d,
        input_wrangler=_max_pool_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is True,
        reason="this aten overload assume return_indices=False",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool2d_with_indices",
        nn_ops.aten_max_pool2d_with_indices,
        input_wrangler=_max_pool_input_wrangler,
    ).skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is False,
        reason="this aten overload assume return_indices=True",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool3d",
        nn_ops.aten_max_pool3d,
        input_wrangler=_max_pool_input_wrangler,
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is True,
        reason="this aten overload assume return_indices=False",
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool3d_with_indices",
        nn_ops.aten_max_pool3d_with_indices,
        input_wrangler=_max_pool_input_wrangler,
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("return_indices") is False,
        reason="this aten overload assume return_indices=True",
    ),
    TorchLibOpInfo(
        "nn.functional.scaled_dot_product_attention",
        nn_ops.aten_scaled_dot_product_attention,
        tolerance={torch.float32: (3e-4, 1.5e-5)},
    )
    .skip(
        matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask")) is not None
        and attn_mask.dtype == torch.bool,
        reason="this overload takes a non-boolean mask",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    TorchLibOpInfo(
        "ops.aten._scaled_dot_product_flash_attention",
        nn_ops.aten__scaled_dot_product_flash_attention,
        tolerance={torch.float32: (3e-4, 1.5e-5)},
        # Output[0] is OK, but other outputs just have the same shape with zero values
        nondeterministic=True,
        compare_shape_only_for_output=(1, 2, 3, 4, 5, 6, 7, 8),
    ).skip(
        device_type="cpu",
        reason="_scaled_dot_product_flash_attention only supports CUDA",
    ),
    TorchLibOpInfo(
        "ops.aten._scaled_dot_product_efficient_attention",
        nn_ops.aten__scaled_dot_product_efficient_attention,
        tolerance={torch.float32: (3e-4, 1.5e-5)},
        # Output[0] is OK, but other outputs just have the same shape with zero values
        nondeterministic=True,
        compare_shape_only_for_output=(1, 2, 3),
    ).skip(
        enabled_if=not torch.cuda.is_available(),
        reason="_scaled_dot_product_efficient_attention only supports CUDA",
    ),
    TorchLibOpInfo(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        nn_ops.aten_scaled_dot_product_attention_bool_mask,
        tolerance={torch.float32: (3e-4, 1.5e-5)},
    )
    .skip(
        matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask")) is not None
        and attn_mask.dtype != torch.bool,
        reason="this overload takes a boolean mask",
    )
    .skip(
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    )
    .xfail(
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_bilinear2d.default",
        nn_ops.aten_upsample_bilinear2d,
    ).xfail(
        matcher=lambda sample: sample.args[1] is False
        and sample.kwargs.get("scales_h") is not None,
        reason="fixme: align_corners=False output mismatch when scales are provided",
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_bilinear2d.vec",
        nn_ops.aten_upsample_bilinear2d_vec,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_bicubic2d.default",
        nn_ops.aten_upsample_bicubic2d,
    ).xfail(
        matcher=lambda sample: sample.args[1] is False
        and sample.kwargs.get("scales_h") is not None,
        reason="fixme: align_corners=False output mismatch when scales are provided",
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_bicubic2d.vec",
        nn_ops.aten_upsample_bicubic2d_vec,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_linear1d",
        nn_ops.aten_upsample_linear1d,
    ).xfail(
        matcher=lambda sample: sample.args[1] is False
        and sample.kwargs.get("scales") is not None,
        reason="fixme: align_corners=False output mismatch when scales are provided",
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest1d",
        nn_ops.aten_upsample_nearest1d,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest1d.vec",
        nn_ops.aten_upsample_nearestnd_vec,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest2d",
        nn_ops.aten_upsample_nearest2d,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest2d.vec",
        nn_ops.aten_upsample_nearestnd_vec,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest3d",
        nn_ops.aten_upsample_nearest3d,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_nearest3d.vec",
        nn_ops.aten_upsample_nearestnd_vec,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_trilinear3d.default",
        nn_ops.aten_upsample_trilinear3d,
    ),
    TorchLibOpInfo(
        "ops.aten.upsample_trilinear3d.vec",
        nn_ops.aten_upsample_trilinear3d_vec,
    ),
    TorchLibOpInfo("ones_like", core_ops.aten_ones_like),
    TorchLibOpInfo(
        "roll",
        core_ops.aten_roll,
        input_wrangler=_roll_input_wrangler,
    ),
    TorchLibOpInfo(
        "roll",
        core_ops.aten_roll_complex,
        input_wrangler=_roll_input_wrangler,
        complex=True,
    ),
    TorchLibOpInfo(
        "scatter_reduce",
        core_ops.aten_scatter_reduce,
        input_wrangler=_scatter_reduce_input_wrangler,
    )
    .xfail(
        variant_name="mean",
        reason="ONNX doesn't support reduce='mean' option",
    )
    .skip(
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    )
    .xfail(
        variant_name="amax",
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'max'",
    )
    .xfail(
        variant_name="amin",
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'min'",
    )
    .xfail(
        variant_name="prod",
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'prod'",
    )
    .xfail(
        variant_name="sum",
        reason="fixme: MLFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'add'",
    ),
    TorchLibOpInfo("ops.aten.slice_scatter", core_ops.aten_slice_scatter),
    TorchLibOpInfo("slice", core_ops.aten_slice),
    TorchLibOpInfo(
        "sum",
        core_ops.aten_sum_dim_IntList,
        input_wrangler=_sum_input_wrangler,
    ).xfail(
        dtypes=(torch.int32,),
        reason="fixme: torch.sum uses int64 as the accumulator for int32 inputs",
    ),
    TorchLibOpInfo(
        "ops.aten.tensor.bool", core_ops.aten_tensor_bool
    ),  # Custom from extra_opinfo
    TorchLibOpInfo(
        "ops.aten.tensor.float",
        core_ops.aten_tensor_float,  # Custom from extra_opinfo
    ),
    TorchLibOpInfo(
        "ops.aten.tensor.int", core_ops.aten_tensor_int
    ),  # Custom from extra_opinfo
    TorchLibOpInfo("transpose", core_ops.aten_transpose),
    TorchLibOpInfo("transpose", core_ops.aten_transpose_complex, complex=True),
    TorchLibOpInfo(
        "ops.prims.var.default",
        prims_ops.prims_var,
        tolerance={torch.float16: (1e-3, 5e-2)},
    ),
    TorchLibOpInfo("zeros_like", core_ops.aten_zeros_like),
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
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB"
assert NONDETERMINISTIC_OPS.issubset(
    TESTED_OPS
), f"{NONDETERMINISTIC_OPS - TESTED_OPS} not in TESTED_OPS"
