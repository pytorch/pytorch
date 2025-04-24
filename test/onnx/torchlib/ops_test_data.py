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
# flake8: noqa

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Any, Callable, Collection, Optional
from typing_extensions import Self

import numpy as np
import ops_test_common

import torch
from torch.onnx._internal.exporter._torchlib.ops import core as core_ops
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import definitions as opinfo_definitions


# Create a copy of the op_db to modify
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

# Append extra op_db into the op database for testing
OPS_DB.extend(opinfo_definitions.signal.op_db)


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
    TorchLibOpInfo("add", core_ops.aten_add, tolerance={torch.float16: (1e-3, 1e-3)}),
    TorchLibOpInfo("add", core_ops.aten_add_complex, complex=True),
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
assert NONDETERMINISTIC_OPS.issubset(TESTED_OPS), (
    f"{NONDETERMINISTIC_OPS - TESTED_OPS} not in TESTED_OPS"
)
