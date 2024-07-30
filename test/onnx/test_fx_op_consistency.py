# Owner(s): ["module: onnx"]

"""Test consistency between the output values of torch.onnx FX exported operators
and torch operators given the same inputs.

Usage:

    1. Test all operators:

    pytest test/onnx/test_fx_op_consistency.py

    2. To run tests on a specific operator (e.g. torch.ceil):

    pytest test/onnx/test_fx_op_consistency.py -k ceil
    pytest test/onnx/test_fx_op_consistency.py -k nn_functional_scaled_dot_product_attention

    3. Set `CREATE_REPRODUCTION_REPORT=1` to create markdown files for reproduction of errors. E.g.

    CREATE_REPRODUCTION_REPORT=1 python -m pytest test/onnx/test_fx_op_consistency.py -k div_mode_int

    NOTE: Read more on Running and writing tests:
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

Note:

    1. Please make sure pytest-subtests is installed. Otherwise, the sub-tests will be ignored.

    2. Install pytest-xdist to run tests in parallel if runng all tests is the goal.

    3. When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES and
    TESTED_OPS lists. See "Modify this section"

"""

from __future__ import annotations

import copy
import itertools
import os
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import error_reproduction

import onnx_test_common

import parameterized
import pytest
import pytorch_test_common
from onnx_test_common import skip, skip_slow, xfail

import torch
from torch.onnx._internal.diagnostics import _rules
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.opinfo import core as opinfo_core  # noqa: TCH001


# NOTE: For ATen signature modifications that will break ONNX export,
# use **xfail_torchlib_forward_compatibility** and **skip_torchlib_forward_compatibility** instead of xfail or skip
# to make the signal apparent for maintainers.
def xfail_torchlib_forward_compatibility(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    github_issue: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], bool]] = None,
    enabled_if: bool = True,
):
    """Prefer using this (xfail) over skip when possible.

    Only skip when the test is not failing consistently.
    """
    return xfail(
        op_name,
        variant_name=variant_name,
        reason=f"{reason}. GitHub Issue: {github_issue}",
        opsets=opsets,
        dtypes=dtypes,
        matcher=matcher,
        enabled_if=enabled_if,
    )


def skip_torchlib_forward_compatibility(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    github_issue: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
):
    """Prefer using xfail_torchlib_forward_compatibility over this (skip) when possible.

    Only skip when the test is not failing consistently.
    """
    return skip(
        op_name,
        variant_name=variant_name,
        reason=f"{reason}. GitHub Issue: {github_issue}",
        opsets=opsets,
        dtypes=dtypes,
        matcher=matcher,
        enabled_if=enabled_if,
    )


# fmt: off
# Turn off black formatting to keep the list compact

# Expected failures for onnx export.
# The list should be sorted alphabetically by op name.
# Q: When should I use fixme vs vs skip vs xfail?
# A: Prefer xfail over skip when possible.
#     2a. If a test is now failing because of xpass, because some previous errors
#     are now fixed, removed the corresponding xfail.
#     2b. If a test is not failing consistently, use skip.
# NOTE: EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES only supports dtypes. If a matcher or model_type
# is needed, use the SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE list further down below.
EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES: Tuple[onnx_test_common.DecorateMeta, ...] = (
    xfail(
        "__getitem__",
        reason="io_adaper doesn't support __getitem__ input slice(0, 3, None)",
    ),
    xfail(
        "__radd__",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Add", "bool"),
    ),
    xfail(
        "__rmatmul__",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "__rpow__",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Pow", "int"),
    ),
    skip(
        "_native_batch_norm_legit",
        reason=onnx_test_common.reason_onnx_script_does_not_support("cpu is not supported: \
            https://github.com/microsoft/onnxscript/pull/1289")
    ),
    skip(
        "_batch_norm_with_update",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch and type error",
    ),
    xfail(
        "_softmax_backward_data",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "_unsafe_masked_index",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "_unsafe_masked_index",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("_unsafe_masked_index", "complex64"),
    ),
    xfail(
        "_unsafe_masked_index_put_accumulate",
        reason="fixme: Status Message: updates tensor should have shape equal to "
               "indices.shape[:-1] + data.shape[indices.shape[-1]:]",
    ),
    xfail(
        "_unsafe_masked_index_put_accumulate",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "add", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Add")
    ),
    xfail(
        "add",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "Add", "int8, int16, uint8 have type issue."
        ),
    ),
    xfail(
        "addbmm",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addbmm", "complex64")
    ),
    xfail(
        "addmm", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Addmm")
    ),
    xfail(
        "addmm",
        variant_name="decomposed",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Addmm")
    ),
    skip(
        "addmm", dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addmm", "complex64 (core dump)")
    ),
    skip(
        "addmm",
        variant_name="decomposed",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addmm", "complex64 (core dump)")
    ),
    xfail(
        "addr",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "Addr", "bool"
        ),
    ),
    xfail(
        "addr",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Addr", "complex64")
    ),
    xfail(
        "alias_copy",
        dtypes=(torch.int8, torch.uint8, torch.int16, torch.float64),
        reason="OnnxExporterError: Failed to export model",
    ),
    xfail(
        "allclose",
        reason=onnx_test_common.reason_dynamo_does_not_support("Allclose")
    ),
    xfail(
        "amax",
        dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMin", "bool, int16"),
    ),
    xfail(
        "amin", dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_dynamo_does_not_support("ReduceMin", "bool, int16")
    ),
    xfail(
        "aminmax",
        dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES),
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMin", "bool, int16"),
    ),
    xfail(
        "arange",
        dtypes=(torch.uint8,),
        reason=onnx_test_common.reason_onnx_script_does_not_support("Arange", "uint8, int8"),
    ),
    xfail(
        "arange",
        dtypes=(torch.int16, torch.int32),
        reason="AssertionError: The values for attribute 'shape' do not match",
    ),
    xfail(
        "argmax",
        dtypes=(
            torch.int16,
            torch.int64,
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ArgMax", "int16, int64"
        ),
    ),
    xfail(
        "argmin",
        dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int64,
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ArgMin", "uint8, int8, int16, int64"
        ),
    ),
    xfail(
        "argwhere",
        reason="fixme: Assertion error: result mismatch",
    ),
    skip(
        "as_strided",
        variant_name="partial_views",
        reason="ONNX doesn't have partial view for tensor; [PostInline][ORT] segfaults",
    ),
    xfail(
        "atan2",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "baddbmm",
        dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Matmul", "uint8, int8, int16"
        ),
    ),
    xfail(
        "baddbmm",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("baddbmm", "complex64")
    ),
    xfail(
        "bernoulli",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "bfloat16",
        reason="fixme: ORT errors with RuntimeError: No corresponding Numpy type for Tensor Type.",
    ),
    xfail(
        "bincount",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.bincount.default"),
    ),
    xfail(
        "block_diag",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Block_diag", "complex"),
    ),
    xfail(
        "bmm",
        dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Matmul", "uint8, int8, int16"
        ),
    ),
    xfail(
        "broadcast_shapes",
        reason=onnx_test_common.reason_dynamo_does_not_support("output is int"),
    ),
    xfail(
        "cauchy",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    skip(
        "ceil", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Ceil", "bool and int")
    ),
    xfail(
        "chalf",
        reason="fixme: ONNX shape type inference error: Invalid tensor data type 0."
    ),
    xfail(
        "chunk",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Chunk", "uint8, int8, int16"
        ),
    ),
    xfail(
        "clamp",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"
        ),
    ),
    xfail(
        "clamp_max", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Clamp_max", "bool")
    ),
    xfail(
        "clamp_max",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"
        ),
    ),
    xfail(
        "clamp_min",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Max", "uint8, int8, int16"
        ),
    ),
    xfail(
        "clamp_min", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Clamp_min", "bool")
    ),
    xfail(
        "constant_pad_nd",
        dtypes=(torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Constant_pad_nd", "int16"
        ),
    ),
    xfail(
        "constant_pad_nd",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "Constant_pad_nd", "complex64"
        ),
    ),
    xfail(
        "corrcoef",
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "aten.equal.default"
        ),
    ),
    xfail(
        "cov",
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "aten.equal.default"
        ),
    ),
    xfail(
        "cumsum", dtypes=onnx_test_common.BOOL_TYPES + (torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("Cumsum", "bool, uint8, int8, int16")
    ),
    xfail(
        "combinations",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.masked.select"),
    ),
    xfail(
        "diag",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Diagonal", "bool"),
    ),
    xfail(
        "diagonal_copy",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Diagonal", "bool"),
    ),
    xfail(
        "dot", dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("MatMul", "uint8, int8, int16")
    ),
    skip(
        "dot",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("Dot", "complex64(core dump)"),
    ),
    xfail(
        "empty",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."
    ),
    xfail(
        "empty_strided",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "eq",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Equal", "uint8, int8, int16"),
    ),
    xfail(
        "equal",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")
    ),
    xfail(
        "exponential",
        reason=onnx_test_common.reason_dynamo_does_not_support("exponential"),
    ),
    xfail(
        "fft.fft",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.fft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.fftn",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.ifft",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.ifft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.ifftn",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.irfft",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.irfft2",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "fft.irfftn",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    xfail(
        "fft.rfft",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    xfail(
        "fft.rfftn",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    xfail(
        "fft.rfft2",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._fft_r2c.default"),
    ),
    xfail(
        "floor",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "bool, int"),
    ),
    xfail(
        "floor_divide",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "bool, int"),
    ),
    xfail(
        "full",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("full", "complex64")
    ),
    xfail(
        "full_like",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("full_like", "complex64")
    ),
    xfail(
        "gather",
        reason="GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'"
    ),
    xfail(
        "geometric",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "heaviside",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Heaviside", "bool"),
    ),
    xfail(
        "index_add",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterND", "int64, int32, bool"),
    ),
    xfail(
        "index_fill",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("index_fill", "complex64")
    ),
    xfail(
        "index_fill",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES,
        reason="fixme: Constant input list has None. ONNXScript does not support None in constant list."
    ),
    xfail(
        "index_put",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),
        reason=onnx_test_common.reason_onnx_script_does_not_support("index_put", "bool"),
    ),
    xfail(
        "index_put",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_script_does_not_support("Add", "int8, int16"),
    ),
    xfail(
        "index_put",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterND", "float16"),
    ),
    xfail(
        "isnan",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("IsNaN", "int, bool"),
    ),
    xfail(
        "istft",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    xfail(
        "item",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "lerp",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("lerp", "complex64")
    ),
    xfail(
        "linalg.lstsq",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.linalg_lstsq.default"),
    ),
    xfail(
        "linalg.lstsq",
        variant_name="grad_oriented",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.linalg_lstsq.default"),
    ),
    xfail(
        "linalg.matrix_power",
        reason="fixme: The values for attribute 'shape' do not match: torch.Size([2, 2]) != torch.Size([2, 2, 2])."
    ),
    xfail(
        "linalg.norm",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "linalg.norm",
        variant_name="subgradients_at_zero",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "linalg.vecdot",
        reason="fixme: Assertion error: result shape mismatch",
    ),
    xfail(
        "linspace",
        dtypes=(torch.int64, torch.int32,),
        reason="fixme: Results do not match with PyTorch. https://github.com/microsoft/onnxscript/issues/854",
    ),
    xfail(
        "linspace",
        variant_name="tensor_overload",
        dtypes=(torch.int64, torch.int32,),
        reason="fixme: Results do not match with PyTorch. https://github.com/microsoft/onnxscript/issues/854",
    ),
    xfail(
        "linspace",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("linspace", "complex64")
    ),
    xfail(
        "linspace",
        variant_name="tensor_overload",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("linspace", "complex64")
    ),
    xfail(
        "log_normal",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "log_softmax",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "log_softmax",
        variant_name="with_dtype",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "logical_and",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("And", "float, int"),
    ),
    xfail(
        "logical_not",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Not", "float, int"),
    ),
    xfail(
        "logical_or",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Or", "float, int"),
    ),
    xfail(
        "logical_xor",
        dtypes=onnx_test_common.FLOAT_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Xor", "float, int"),
    ),
    skip(
        "masked.logsumexp",
        reason="fixme: https://github.com/onnx/onnx/issues/4986",
    ),
    xfail(
        "masked.amax",
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "masked.amin",
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "masked.argmin",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES + (torch.int64,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "masked.argmax",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.FLOAT_TYPES + (torch.int64,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "masked_fill",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "masked.sum",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "masked.log_softmax",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "masked.mean",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("ReduceMean", "bool"),
    ),
    xfail(
        "masked.norm",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "masked.prod",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "masked_select",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.masked_select.default"),
    ),
    xfail(
        "max",
        variant_name="reduction_no_dim",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMax", "bool"),
    ),
    xfail(
        "max",
        variant_name="reduction_with_dim",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMax", "bool"),
    ),
    xfail(
        "max",
        variant_name="reduction_with_dim",
        dtypes=(torch.int64,),
        reason="https://github.com/onnx/onnx/issues/4986",
    ),
    xfail(
        "min",
        variant_name="reduction_no_dim",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMin", "bool"),
    ),
    xfail(
        "min",
        variant_name="reduction_with_dim",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int64,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ReduceMin", "bool"),
    ),
    skip(
        "mm",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("MM", "complex64(core dump)"),
    ),
    xfail(
        "multinomial",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nanquantile",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")
    ),
    xfail(
        "nansum",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("IsNaN", "int, bool"),
    ),
    xfail(
        "narrow",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    skip(
        "native_batch_norm",
        reason=onnx_test_common.reason_onnx_script_does_not_support("cpu is not supported: \
            https://github.com/microsoft/onnxscript/pull/1289")
    ),
    xfail(
        "native_layer_norm",
        dtypes=(torch.float16,),
        reason="fixme: ORT optimizer error: https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "new_full",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason=onnx_test_common.reason_dynamo_does_not_support("new_full", "complex64")
    ),
    xfail(
        "nn.functional.adaptive_avg_pool2d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("RecursionError: \
            maximum recursion depth exceeded while calling a Python object"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool3d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten._adaptive_avg_pool3d.default"),
    ),
    xfail(
        "nn.functional.alpha_dropout",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.avg_pool1d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.avg_pool2d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.avg_pool3d",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("AveragePool", "int"),
    ),
    xfail(
        "nn.functional.batch_norm",
        dtypes=(torch.float16,),
        reason="fixme: https://github.com/microsoft/onnxscript/issues/1270",
    ),
    xfail(
        "nn.functional.conv_transpose1d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
    ),
    xfail(
        "nn.functional.conv_transpose2d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
    ),
    xfail(
        "nn.functional.conv_transpose3d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
    ),
    skip(
        "nn.functional.conv_transpose1d",
        reason="fixme: Assertion error: result mismatch",
    ),
    skip(
        "nn.functional.conv_transpose2d",
        reason="fixme: Assertion error: result mismatch",
    ),
    skip(
        "nn.functional.conv_transpose3d",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.conv1d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv1d", "int64"),
    ),
    xfail(
        "nn.functional.conv2d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv2d", "int64"),
    ),
    xfail(
        "nn.functional.conv2d",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.conv3d",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_does_not_support("Conv3d", "int64"),
    ),
    xfail(
        "nn.functional.conv3d",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.cosine_embedding_loss",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("CosineEmbeddingLoss", "bool"),
    ),
    xfail(
        "nn.functional.ctc_loss",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.ctc_loss.default"),
    ),
    xfail(
        "nn.functional.dropout",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.dropout2d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.dropout3d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.feature_alpha_dropout",
        variant_name="with_train",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.feature_alpha_dropout",
        variant_name="without_train",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.fractional_max_pool2d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.fractional_max_pool3d",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.gaussian_nll_loss",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.gaussian_nll_loss"),
    ),
    xfail(
        "nn.functional.grid_sample",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.group_norm",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("GroupNormalization", "float16"),
    ),
    xfail(
        "nn.functional.local_response_norm",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("avgpool", "int64"),
    ),
    xfail(
        "nn.functional.linear",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Gemm", "int"),
    ),
    xfail(
        "nn.functional.max_pool2d",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Max_pool2d"),
    ),
    xfail(
        "nn.functional.max_pool3d",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Max_pool3d"),
    ),
    xfail(
        "nn.functional.multi_head_attention_forward",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.one_hot",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    xfail(
        "nn.functional.pad",
        variant_name="replicate",
        reason="fixme: ORT error: padding size",
    ),
    xfail(
        "nn.functional.pad",
        variant_name="replicate_negative",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.pad",
        variant_name="reflect",
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nn.functional.pixel_shuffle",
        dtypes=(torch.int32, torch.int64) + onnx_test_common.BOOL_TYPES,
        reason="fixme: ONNX Runtime does not support int32/64 inputs",
    ),
    xfail(
        "nn.functional.pixel_unshuffle",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.pixel_unshuffle.default"),
    ),
    xfail(
        "nn.functional.poisson_nll_loss",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason="fixme: result mismatch with NaN.",
    ),
    xfail(
        "nn.functional.rrelu",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "nn.functional.rrelu",
        dtypes=(torch.int64,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Relu", "int64"),
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    ),
    xfail(
        "nn.functional.scaled_dot_product_attention",
        dtypes=(torch.float16,),
        reason="fixme: ORT failed. https://github.com/microsoft/onnxruntime/issues/16438",
    ),
    xfail(
        "nn.functional.selu",
        reason="fixme: nn.functional.selu is not in torch._decomp.decomposition_table",
    ),
    xfail(
        "nn.functional.soft_margin_loss",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "nonzero",
        dtypes=(torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("NonZero", "int8, int16"),
    ),
    xfail(
        "normal",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "normal",
        variant_name="in_place",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "normal",
        variant_name="number_mean",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "ones",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."
    ),
    xfail(
        "pca_lowrank",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "quantile",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.equal.default")
    ),
    xfail(
        "rand_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "randint",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "randint_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "randn",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "randn_like",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "resize_",
        reason=onnx_test_common.reason_dynamo_does_not_support("resize_as_")
    ),
    xfail(
        "resize_as_",
        reason=onnx_test_common.reason_dynamo_does_not_support("resize_as_")
    ),
    xfail(
        "round",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Round", "int"),
    ),
    xfail(
        "rsub",
        dtypes=(torch.uint8, torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Mul", "uint8, int8, int16"
        ),
    ),
    xfail(
        "scatter_add",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=sum", "float16"),
    ),
    xfail(
        "scatter_reduce",
        variant_name="sum",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=sum", "float16"),
    ),
    xfail(
        "scatter_reduce",
        variant_name="prod",
        dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=prod", "float16"),
    ),
    xfail(
        "scatter_reduce",
        variant_name="amin",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=amin", "float16"),
    ),
    xfail(
        "scatter_reduce",
        variant_name="amax",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("ScatterElements reduction=amax", "float16"),
    ),
    xfail(
        "scatter_reduce",
        variant_name="mean",
        reason="ONNX doesn't support reduce='mean' option",
    ),
    xfail(
        "sgn",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Sign", "bool"),
    ),
    xfail(
        "sign",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Sign", "bool"),
    ),
    xfail(
        "signal.windows.kaiser",
        reason=onnx_test_common.reason_dynamo_does_not_support("functionalization"),
    ),
    xfail(
        "softmax",
        dtypes=(torch.float16,),
        reason="ORT error: https://github.com/microsoft/onnxruntime/issues/16438"
    ),
    xfail(
        "sparse.mm",
        variant_name="reduce",
        reason=onnx_test_common.reason_dynamo_does_not_support("InternalTorchDynamoError: Sparse CSR tensors do not have strides"),
    ),
    xfail(
        "sparse.sampled_addmm",
        reason=onnx_test_common.reason_dynamo_does_not_support("InternalTorchDynamoError: Sparse CSR tensors do not have strides"),
    ),
    xfail(
        "special.erfcx",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Erf", "int, bool"),
    ),
    xfail(
        "special.erfcx",
        dtypes=onnx_test_common.FLOAT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Erfcx"),
    ),
    xfail(
        "special.log_ndtr",
        dtypes=onnx_test_common.INT_TYPES + onnx_test_common.FLOAT_TYPES,
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "special.ndtr",
        dtypes=(torch.float16,),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "square",
        dtypes=(torch.int8, torch.uint8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Pow", "int8, uint8, int16"),
    ),
    xfail(
        "squeeze",
        variant_name="multiple",
        reason="fixme: https://github.com/microsoft/onnxscript/issues/1264",
    ),
    xfail(
        "svd_lowrank",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "stft",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten._fft_r2c.default"),
    ),
    xfail(
        "sub",
        dtypes=(torch.uint8, torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Mul", "uint8, int8, int16"
        ),
    ),
    xfail(
        "take",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    xfail(
        "tensor_split",
        reason=onnx_test_common.reason_dynamo_does_not_support("data-dependent"),
    ),
    xfail(
        "topk",
        dtypes=(torch.int64, torch.int32, torch.float16),
        reason="fixme: Assertion error: result mismatch",
    ),
    xfail(
        "tril",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int32,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("trilu", "bool, int32"),
    ),
    xfail(
        "triu",
        dtypes=onnx_test_common.BOOL_TYPES + (torch.int32,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("trilu", "bool, int32"),
    ),
    xfail(
        "trunc",
        dtypes=onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "int"),
    ),
    xfail(
        "unflatten",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Unflatten")
    ),
    xfail(
        "uniform",
        reason=onnx_test_common.reason_dynamo_does_not_support("wrapper_set_seed"),
    ),
    xfail(
        "unique",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.unique_consecutive.default"),
    ),
    xfail(
        "unique_consecutive",
        reason=onnx_test_common.reason_dynamo_does_not_support("aten.unique_consecutive.default"),
    ),
    xfail(
        "unravel_index",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("Floor", "bool, int"),
    ),
    xfail(
        "unsqueeze_copy",
        reason="OnnxExporterError: Failed to export model",
        dtypes=(torch.int8, torch.uint8, torch.int16),
    ),
    xfail(
        "where",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Where", "bool"),
    ),
    xfail(
        "zeros",
        dtypes=onnx_test_common.COMPLEX_TYPES,
        reason="fixme: kwargs dtpye=complex64 is not supported in ONNX."
    ),
    # SLOW TESTS (All are xfails if we run them)
    # TODO: https://github.com/pytorch/pytorch/issues/117118
    skip_slow(
        "cdist",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "histogram",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "histogramdd",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "linalg.lu_solve",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "linalg.solve_triangular",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "linalg.svd",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "logspace",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "logspace",
        variant_name="tensor_overload",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "max_pool2d_with_indices_backward",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.interpolate",
        variant_name="bicubic",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_unpool1d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_unpool2d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_unpool3d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_pool1d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_pool2d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.max_pool3d",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "nn.functional.unfold",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "ormqr",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "searchsorted",
        reason="fixme: Test sets are too many.",
    ),
    skip_slow(
        "svd",
        reason="fixme: Test sets are too many.",
    ),
)
# fmt: on

# NOTE: The xfail and skip with a matcher function or model_type should be
# at under the `SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE` section.
SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE: tuple[
    onnx_test_common.DecorateMeta, ...
] = (
    skip(
        "_native_batch_norm_legit",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    skip(
        "_batch_norm_with_update",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    # TODO: This test currently fails only for certain inputs, e.g. shape([3, 1]).
    # Numerically the ONNX program is correct, but the output shapes for `save_mean`
    # and `save_var` were tensor(-2.1268) instead of the correct tensor([-2.1268])
    # for example.
    skip(
        "_batch_norm_with_update",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
        reason="not supported yet",
    ),
    xfail(
        "addmm",  # xfail can't only use dtypes to catch all cases
        matcher=lambda sample: sample.input.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Gemm", "uint8, int8, int16, int32, int64"
        ),
    ),
    xfail(
        "addmm",
        matcher=lambda sample: sample.args[0].numel() == 0,
        reason="ONNX Runtime does not support empty tensors multiplication",
    ),
    xfail(
        "addmm",
        variant_name="decomposed",
        matcher=lambda sample: sample.args[0].numel() == 0,
        reason="ONNX Runtime does not support empty tensors multiplication",
    ),
    xfail(
        "amax",
        matcher=lambda sample: len(sample.input.shape) == 0
        and (sample.kwargs.get("dim") is not None and sample.kwargs.get("dim") != ()),
        reason="Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    xfail(
        "amin",
        matcher=lambda sample: len(sample.input.shape) == 0
        and (sample.kwargs.get("dim") is not None and sample.kwargs.get("dim") != ()),
        reason="Op (ReduceMin) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    xfail(
        "aminmax",
        matcher=lambda sample: len(sample.input.shape) == 0
        and sample.kwargs.get("dim") is not None,
        reason="Op (ReduceMin) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    skip(
        "cat",
        matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
        reason="core dump - cat does not support zero-dim tensors yet",
    ),
    xfail(
        "index_add",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ScatterND", "0-D tensor"
        ),
    ),
    xfail(
        "index_add",
        matcher=lambda sample: isinstance(sample.args[0], int) and sample.args[0] == -1,
        reason="fixme: aten::index_put indices contains None when dim is -1",
    ),
    xfail(
        "index_copy",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "ScatterND", "0-D tensor"
        ),
    ),
    xfail(
        "index_copy",
        matcher=lambda sample: isinstance(sample.args[0], int) and sample.args[0] == -1,
        reason="fixme: aten::index_put indices contains None when dim is -1",
    ),
    xfail(
        "index_put",
        matcher=lambda sample: (sample.args[0][0].dtype == torch.bool)
        and (sample.kwargs.get("accumulate") is False),
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "https://github.com/pytorch/pytorch/issues/101150"
        ),
    ),
    skip(
        "linalg.multi_dot",
        matcher=lambda sample: sum(torch.numel(input) for input in sample.input) == 0,
        reason="fixme: Undefined",
    ),
    skip(
        "log_softmax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    ),
    skip(
        "log_softmax",
        variant_name="with_dtype",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    ),
    skip(
        "masked.log_softmax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    ),
    skip(
        "matmul",
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
    ),
    skip(
        "mm",
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
    ),
    xfail(
        "native_batch_norm",
        matcher=lambda sample: sample.args[-3] is True
        and any(arg is not None for arg in sample.args[2:4]),
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    xfail(
        "nn.functional.avg_pool1d",
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
    xfail(
        "nn.functional.avg_pool2d",
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)
        or (sample.kwargs.get("divisor_override") is not None),
        reason="ONNX doesn't support divisor_override argument",
    ),
    xfail(
        "nn.functional.avg_pool3d",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True,
        reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19",
    ),
    xfail(
        "nn.functional.avg_pool3d",
        matcher=lambda sample: (len(sample.args) > 5 and sample.args[5] is not None)
        or (sample.kwargs.get("divisor_override") is not None),
        reason="ONNX doesn't support divisor_override argument",
    ),
    xfail(
        "nn.functional.batch_norm",
        matcher=lambda sample: sample.kwargs.get("training") is True
        and any(arg is not None for arg in sample.args[2:4]),
        reason="Flaky failure: https://github.com/pytorch/pytorch/issues/115106",
    ),
    xfail(
        "nn.functional.conv2d",
        matcher=lambda sample: sample.kwargs.get("padding") == "valid",
        reason="fixme: https://github.com/pytorch/pytorch/issues/117054",
    ),
    xfail(
        "nn.functional.conv3d",
        matcher=lambda sample: sample.kwargs.get("padding") == "valid",
        reason="fixme: https://github.com/pytorch/pytorch/issues/117054",
    ),
    skip(
        "nn.functional.cross_entropy",
        matcher=lambda sample: not isinstance(sample.kwargs.get("weight"), int),
        reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[weight] is int type",
    ),
    xfail(
        "nn.functional.embedding",
        matcher=lambda sample: sample.kwargs.get("max_norm") is not None,
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="https://github.com/pytorch/pytorch/issues/115106",
    ),
    skip_torchlib_forward_compatibility(
        "nn.functional.embedding_bag",
        matcher=lambda sample: sample.kwargs.get("padding_idx") is not None or True,
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "'padding_idx' overload for _embedding_bag and _embedding_bag_forward_only. "
            "'padding_idx=-1' is emitted for aten op when 'padding_idx' is not provided"
        ),
        github_issue="https://github.com/microsoft/onnxscript/issues/1056",
    ),
    xfail(
        "nn.functional.group_norm",
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Reshape", "empty tensor"
        ),
    ),
    xfail(
        "nn.functional.instance_norm",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        matcher=lambda sample: sample.kwargs.get("running_mean") is not None,
        reason="fixme: KeyError: 'self___kwargs__running_mean'",
    ),
    xfail(
        "nn.functional.max_pool3d",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    ),
    xfail(
        "nn.functional.pixel_shuffle",
        matcher=lambda sample: sample.input.numel() == 0,
        reason="fixme: ORT does not support empty tensor as input",
    ),
    xfail(
        "nonzero",
        matcher=lambda sample: len(sample.input.shape) == 0
        and sample.kwargs.get("as_tuple", False) is False,
        reason="Output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0]).",
        model_type=pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
    ),
    xfail(
        "scatter_add",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch",
    ),
    skip(
        "scatter_reduce",
        variant_name="amax",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "scatter_reduce",
        variant_name="amin",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "scatter_reduce",
        variant_name="prod",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "scatter_reduce",
        variant_name="sum",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "softmax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: LogSoftMax does not support empty tensor as input",
    ),
    xfail(
        "unflatten",
        reason="Logic not implemented for size 0 inputs in op.Reshape",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
    ),
    skip(
        "signal.windows.hamming",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    skip(
        "signal.windows.general_hamming",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    skip(
        "signal.windows.blackman",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    skip(
        "signal.windows.general_cosine",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    skip(
        "signal.windows.hann",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
    skip(
        "signal.windows.nuttall",
        model_type=pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="does not match node name",
    ),
)

OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(
    meta.op_name for meta in SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE
)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)


def _torch_size_flatten_spec(d: List[Any], spec: Any) -> List[Any]:
    return [d[i] for i in range(spec.num_children)]


torch.fx._pytree.register_pytree_flatten_spec(
    torch.Size,
    _torch_size_flatten_spec,
)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


def _should_skip_xfail_test_sample(
    op_name: str,
    variant_test_name: str,
    sample,
    model_type: pytorch_test_common.TorchModelType,
) -> Tuple[Optional[str], Optional[str]]:
    """Check if the test sample should be skipped or xfailed.

    If the xfail/skip decorator meta is matched with its op_name and model_type,
    return the test_behavior and reason. Otherwise, return None, None. Note that
    if the matcher is None, the test is decorator_meta is meant to skip/xfail all model types.

    Args:
        op_name: The name of the op.
        sample: The test sample.
        model_type: The model type of the test.

    Returns:
        A tuple of (test_behavior, reason). test_behavior is either "skip" or "xfail".
        reason is the reason for the test_behavior.
    """

    if op_name not in OP_WITH_SKIPPED_XFAIL_SUBTESTS:
        return None, None
    for decorator_meta in SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE:
        # Linear search on ops_test_data.SKIP_XFAIL_SUBTESTS_WITH_MATCHER_AND_MODEL_TYPE. That's fine because the list is small.
        # NOTE: If model_type is None, the test is decorator_meta is meant to skip/xfail all model types.
        if (
            decorator_meta.op_name == op_name
            and decorator_meta.variant_name == variant_test_name
        ) and (
            model_type == decorator_meta.model_type or decorator_meta.model_type is None
        ):
            if decorator_meta.matcher is None and decorator_meta.model_type is None:
                raise TypeError(
                    "Either Matcher or model_type must be defined in sub xfail and skip."
                )
            if decorator_meta.matcher is not None and decorator_meta.matcher(sample):
                return decorator_meta.test_behavior, decorator_meta.reason
            elif decorator_meta.matcher is None:
                # xfail/skip the whole test of the model type without matcher
                return decorator_meta.test_behavior, decorator_meta.reason
    return None, None


def _compare_onnx_and_torch_exported_program(
    torch_exported_program,
    onnx_exported_program,
    input_args,
    input_kwargs=None,
    test_name=None,
    sample_num=None,
    sample_kwargs=None,
    rtol=1e-03,
    atol=1e-07,
    only_check_shape=False,
):
    # avoid mutable default argument
    if input_kwargs is None:
        input_kwargs = {}

    # NOTE: ONNXProgram holds a reference (not copy) to the original ref_model, including its state_dict.
    # Thus, ONNXProgram() must run before ref_model() to prevent ref_model.forward() from changing the state_dict.
    # Otherwise, the ref_model can change buffers on state_dict which would be used by ONNXProgram.__call__()
    onnx_outputs = onnx_exported_program(*input_args, **input_kwargs)
    if isinstance(torch_exported_program, torch.export.ExportedProgram):
        torch_outputs = torch_exported_program.module()(*input_args, **input_kwargs)
    else:
        torch_outputs = torch_exported_program(*input_args, **input_kwargs)
    torch_outputs_onnx_format = onnx_exported_program.adapt_torch_outputs_to_onnx(
        torch_outputs
    )
    if len(torch_outputs_onnx_format) != len(onnx_outputs):
        raise AssertionError(
            f"Expected {len(torch_outputs_onnx_format)} outputs, got {len(onnx_outputs)}"
        )

    for j, (torch_output, onnx_output) in enumerate(
        zip(torch_outputs_onnx_format, onnx_outputs)
    ):
        if only_check_shape:
            assert torch_output.shape == onnx_output.shape
        else:
            try:
                torch.testing.assert_close(
                    torch.tensor(onnx_output),
                    torch_output,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            except AssertionError as e:
                if os.environ.get("CREATE_REPRODUCTION_REPORT") == "1":
                    error_reproduction.create_mismatch_report(
                        test_name,
                        sample_num,
                        onnx_exported_program.model_proto,
                        input_args,
                        sample_kwargs,
                        torch.tensor(onnx_output),
                        torch_output,
                        e,
                    )
                if len(torch_outputs_onnx_format) > 1:
                    raise AssertionError(f"Output {j} mismatch") from e
                raise


def _run_test_output_match(
    test_suite: onnx_test_common._TestONNXRuntime,
    device: str,
    dtype: torch.dtype,
    op: opinfo_core.OpInfo,
):
    # device is provided by instantiate_device_type_tests, but we only want to run in cpu.
    assert device == "cpu"
    samples = op.sample_inputs(
        device,
        dtype,
        requires_grad=False,
    )
    for i, cpu_sample in enumerate(samples):
        inputs = (cpu_sample.input, *cpu_sample.args)
        # Provide the repr to subtest because tensors are not serializable in parallel test runs

        with test_suite.subTest(
            opset=test_suite.opset_version,
            sample_num=i,
            inputs=repr(inputs),
            kwargs=repr(cpu_sample.kwargs),
        ):
            test_behavior, reason = _should_skip_xfail_test_sample(
                op.name, op.variant_test_name, cpu_sample, test_suite.model_type
            )
            with onnx_test_common.normal_xfail_skip_test_behaviors(
                test_behavior, reason
            ):
                model = SingleOpModel(op.op, cpu_sample.kwargs)
                model.eval()

                if (
                    dtype == torch.float32
                    and op.name in test_suite.fp32_low_precision_dict
                ):
                    rtol = test_suite.fp32_low_precision_dict[op.name][0]
                    atol = test_suite.fp32_low_precision_dict[op.name][1]
                elif dtype == torch.float32:
                    # Relax atol and rtol for float32 based on empirical results
                    rtol = 1e-5
                    atol = 2e-5
                elif (
                    dtype == torch.float16
                    and (op.name, op.variant_test_name)
                    in test_suite.fp16_low_precision_variant_dict
                ):
                    rtol = test_suite.fp16_low_precision_variant_dict[
                        (op.name, op.variant_test_name)
                    ][0]
                    atol = test_suite.fp16_low_precision_variant_dict[
                        (op.name, op.variant_test_name)
                    ][1]
                elif (
                    dtype == torch.float16
                    and op.name in test_suite.fp16_low_precision_dict
                ):
                    rtol = test_suite.fp16_low_precision_dict[op.name][0]
                    atol = test_suite.fp16_low_precision_dict[op.name][1]
                else:
                    rtol = None
                    atol = None

                if (
                    test_suite.model_type
                    == pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM
                ):
                    try:
                        # TODO (tugsbayasgalan) Migrate to pre-dispatch IR
                        # BUG1: python test/onnx/test_fx_op_consistency.py -k test_output_match_triu_cpu_int32
                        # has unexpected success, but don't know how to remove from xfail list
                        # BUG2: User output to_sparse is not in the correct order or is not found in the
                        # exported program's user_output list (https://github.com/pytorch/pytorch/issues/124328)
                        # python test/onnx/test_fx_op_consistency.py -k test_output_match_to_sparse_cpu_float32
                        # BUG3: [ShapeInferenceError] Inference error(s): (op_type:aten_view, node name: aten_view_4):
                        # [ShapeInferenceError]
                        # Inference error(s): (op_type:Reshape, node name: n1): [ShapeInferenceError] Invalid position of 0.
                        # python test/onnx/test_fx_op_consistency.py -k test_output_match_stack_cpu_int32
                        from torch.export import _trace

                        model = _trace._export(model, inputs, pre_dispatch=False)

                    except AssertionError as e:
                        # NOTE: avoid fake_mode detection bug in torch.export.export
                        pytest.xfail(
                            onnx_test_common.reason_dynamo_does_not_support(str(e))
                        )

                try:
                    onnx_program = torch.onnx.dynamo_export(
                        model,
                        *inputs,
                    )
                except torch.onnx.OnnxExporterError as e:
                    # NOTE: If the model has unsupported nodes, we will skip the test
                    # with non-strict xfail. Otherwise, we will raise the error.
                    if hasattr(
                        e.__cause__, "diagnostic"
                    ) and e.__cause__.diagnostic.rule in (
                        _rules._POERules.no_symbolic_function_for_call_function,
                        _rules._POERules.unsupported_fx_node_analysis,
                    ):
                        pytest.xfail(
                            onnx_test_common.reason_onnx_script_does_not_support(str(e))
                        )
                    else:
                        raise e
                _compare_onnx_and_torch_exported_program(
                    model,
                    onnx_program,
                    inputs,
                    test_name=test_suite.id(),
                    sample_num=i,
                    sample_kwargs=cpu_sample.kwargs,
                    rtol=rtol,
                    atol=atol,
                    only_check_shape=(op.name in test_suite.only_shape_check_list),
                )


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (opset for opset in onnx_test_common.FX_TESTED_OPSETS),
            (
                pytorch_test_common.TorchModelType.TORCH_NN_MODULE,
                pytorch_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
            ),
        )
    )
    return {
        "attrs": ["opset_version", "model_type"],
        "input_values": input_values,
    }


def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffixes = []
    for k, v in input_dicts.items():
        suffixes.append(f"{k}_{v}")
    return f"{cls.__name__}_{'_'.join(suffixes)}"


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),
    class_name_func=_parameterize_class_name,
)
class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    opset_version = -1
    op_level_debug: bool = False
    dynamic_shapes: bool = False
    model_type: pytorch_test_common.TorchModelType = (
        pytorch_test_common.TorchModelType.TORCH_NN_MODULE
    )

    # NOTE: Follow torchlib settings in ops_test_data.py
    only_shape_check_list = [
        "empty",
        "empty_like",
        "empty_strided",
        "new_empty",
        "new_empty_strided",
    ]

    fp32_low_precision_dict = {
        "native_layer_norm": [2e-4, 7e-4],
    }

    fp16_low_precision_dict = {
        "addbmm": [2e-1, 2e-2],
        "addcdiv": [3e-2, 1.4e-3],
        "addcmul": [3e-2, 1e-3],
        "addmv": [5e-2, 3e-2],
        "addr": [3e-3, 4e-3],
        "baddbmm": [3e-2, 1e-3],
        "cumulative_trapezoid": [3e-2, 1e-3],
        "cross": [3e-2, 2e-2],
        "diff": [1e-2, 5e-2],
        "div": [5e-3, 1e-3],
        "gradient": [3e-3, 4e-3],
        "linalg.cross": [1e-3, 2e-2],
        "linalg.multi_dot": [3e-2, 1e-3],
        "linalg.vecdot": [1e-2, 2e-2],
        "linspace": [2e-2, 2e-3],
        "masked.std": [2e-2, 2e-3],
        "masked.var": [2e-2, 2e-2],
        "matmul": [2e-2, 6e-2],
        "mv": [9e-3, 1e-5],
        "nn.functional.batch_norm": [3e-2, 1e-3],
        "nn.functional.binary_cross_entropy": [3e-2, 1e-3],
        "nn.functional.binary_cross_entropy_with_logits": [4e-2, 4e-3],
        "nn.functional.cosine_similarity": [3e-2, 1e-3],
        "nn.functional.cosine_embedding_loss": [1e-2, 1e-3],
        "nn.functional.hardsigmoid": [1e-3, 5e-3],
        "nn.functional.hardswish": [1e-3, 5e-3],
        "nn.functional.hinge_embedding_loss": [4e-1, 3e-3],
        "nn.functional.huber_loss": [1e-2, 1e-1],
        "nn.functional.instance_norm": [1e-2, 1e-3],
        "nn.functional.interpolate": [1e-2, 1e-3],
        "nn.functional.kl_div": [2e-3, 2e-4],
        "nn.functional.multilabel_soft_margin_loss": [4e-2, 5e-3],
        "nn.functional.local_response_norm": [1e-2, 5e-3],
        "nn.functional.poisson_nll_loss": [4e-2, 6e-3],
        "nn.functional.nll_loss": [3e-2, 1e-3],
        "nn.functional.triplet_margin_loss": [2e-2, 1e-2],
        "nn.functional.triplet_margin_with_distance_loss": [3e-2, 1e-2],
        "native_batch_norm": [3e-2, 1e-3],
        "norm": [1e-2, 1e-2],
        "dot": [3e-2, 1e-3],
        "logit": [3e-2, 1e-3],
        "rsub": [3e-2, 1e-3],
        "sinc": [2e-1, 6e-4],
        "sub": [3e-2, 1e-3],
        "trapezoid": [1e-3, 7e-3],
        "trapz": [1e-3, 7e-3],
        "vdot": [1e-3, 1e-2],
    }

    fp16_low_precision_variant_dict = {
        ("nn.functional.interpolate", "trilinear"): [3e-2, 3e-3],
        ("nn.functional.interpolate", "linear"): [3e-2, 3e-3],
    }

    @common_device_type.ops(
        [op for op in OPS_DB if op.name in ALL_OPS_IN_DB],
        allowed_dtypes=onnx_test_common.TESTED_DTYPES,
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        """Test the ONNX exporter."""
        _run_test_output_match(self, device, dtype, op)


for opset in onnx_test_common.FX_TESTED_OPSETS:
    for model_type in pytorch_test_common.TorchModelType:
        # The name needs to match the parameterized_class name.
        test_class_name = f"TestOnnxModelOutputConsistency_opset_version_{opset}_model_type_TorchModelType.{model_type.name}"
        onnx_test_common.add_decorate_info(
            OPS_DB,
            test_class_name,
            "test_output_match",
            opset=opset,
            skip_or_xfails=EXPECTED_SKIPS_OR_FAILS_WITH_DTYPES,
        )

        common_device_type.instantiate_device_type_tests(
            globals()[test_class_name], globals(), only_for="cpu"
        )

if __name__ == "__main__":
    common_utils.run_tests()
