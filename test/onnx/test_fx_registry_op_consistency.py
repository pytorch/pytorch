# Owner(s): ["module: onnx"]

"""Test consistency between the output values of torch.onnx FX exported operators
and torch operators given the same inputs.

Usage:

    pytest test/onnx/test_fx_registry_op_consistency.py

    To run tests on a specific operator (e.g. torch.ceil):

    pytest test/onnx/test_fx_registry_op_consistency.py -k ceil
    pytest test/onnx/test_fx_registry_op_consistency.py -k nn_functional_scaled_dot_product_attention

    Read more on Running and writing tests:
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

Note:

    1. Please make sure pytest-subtests is installed. Otherwise, the sub-tests will be ignored.

    2. Install pytest-xdist to run tests in parallel if runng all tests is the goal.

    3. When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS and
    TESTED_OPS lists. See "Modify this section"

"""

from __future__ import annotations

import copy
import contextlib
import itertools
from typing import Any, Callable, Collection, Mapping, Optional, Tuple, Type, Union, Sequence

import onnx_test_common

import parameterized
import pytest

import torch
from onnx_test_common import skip, xfail
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.opinfo import core as opinfo_core
from torch.onnx._internal.fx import analysis, passes, decomposition_table, diagnostics, onnxfunction_dispatcher
from torch.onnx._internal import io_adapter
from torch.onnx._internal.diagnostics import _rules
from torch.onnx._internal.diagnostics import infra

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
EXPECTED_SKIPS_OR_FAILS: Tuple[onnx_test_common.DecorateMeta, ...] = (
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
        "addmm", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Addmm")
    ),
    xfail(
        "allclose", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES + onnx_test_common.FLOAT_TYPES,
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
    skip(
        "as_strided",
        variant_name="partial_views",
        reason="ONNX doesn't have partial view for tensor; [PostInline][ORT] segfaults",
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
    skip(
        "ceil", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Ceil", "bool and int")
    ),
    xfail(
        "chunk", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Chunk", "bool")
    ),
    xfail(
        "chunk",
        dtypes=(torch.uint8, torch.int8, torch.int16, torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Chunk", "uint8, int8, int16, float16"
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
        "cumsum", dtypes=onnx_test_common.BOOL_TYPES + (torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("Cumsum", "bool, uint8, int8, int16")
    ),
    # See https://github.com/pytorch/pytorch/issues/111454
    xfail(
        "cumsum", dtypes=(torch.float16,),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("RUNTIME_EXCEPTION : \
            Exception during initialization: /onnxruntime_src/onnxruntime/core/framework/\
            allocation_planner.cc:230 int& onnxruntime::PlannerImpl::\
            UseCount(onnxruntime::OrtValueIndex) n >= 0 && static_cast<size_t>(n) \
            < ort_value_info_.size() was false.")
    ),
    xfail(
        "cross",
        reason=onnx_test_common.reason_onnx_script_does_not_support("linalg_cross"),
    ),
    xfail(
        "dot", dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("MatMul", "uint8, int8, int16")
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
        "floor",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Floor", "bool, int"),
    ),
    xfail(
        "index_put",
        dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_script_does_not_support("index_put", "bool"),
    ),
    xfail(
        "index_put",
        dtypes=(torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_script_does_not_support("Add", "int8, int16"),
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
        "nn.functional.dropout",
        reason=onnx_test_common.reason_dynamo_does_not_support("Dropout"),
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
        "nonzero",
        dtypes=(torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("NonZero", "int8, int16"),
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
        "square",
        dtypes=(torch.int8, torch.uint8, torch.int16),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support("Pow", "int8, uint8, int16"),
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
        "unflatten", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Unflatten")
    ),
)
# fmt: on

SKIP_XFAIL_SUBTESTS: tuple[onnx_test_common.DecorateMeta, ...] = (
    xfail(
        "addmm",  # xfail can't only use dtypes to catch all cases
        matcher=lambda sample: sample.input.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
        reason=onnx_test_common.reason_onnx_runtime_does_not_support(
            "Gemm", "uint8, int8, int16, int32, int64"
        ),
    ),
    skip(
        "amax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    skip(
        "amin",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0",
    ),
    xfail(
        "arange",
        matcher=lambda sample: not isinstance(sample.input, torch.Tensor),
        reason="torch.export.export does not support non-tensor input (https://github.com/pytorch/pytorch/issues/115110)",
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    ),
    skip(
        "cat",
        matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
        reason="core dump - cat does not support zero-dim tensors yet",
    ),
    xfail(
        "full",
        matcher=lambda sample: not isinstance(sample.input, torch.Tensor),
        reason="torch.export.export does not support non-tensor input (https://github.com/pytorch/pytorch/issues/115110)",
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
    ),
    xfail(
        "index_put",
        matcher=lambda sample: (sample.args[0][0].dtype == torch.bool)
        and (sample.kwargs.get("accumulate") is False),
        reason=onnx_test_common.reason_dynamo_does_not_support(
            "https://github.com/pytorch/pytorch/issues/101150"
        ),
    ),
    xfail(
        "native_batch_norm",
        matcher=lambda sample: sample.args[-3] is True
        and any(arg is not None for arg in sample.args[2:4]),
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
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
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason="Flaky failure: https://github.com/pytorch/pytorch/issues/115106",
    ),
    skip(
        "nn.functional.conv1d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv1d",
    ),
    skip(
        "nn.functional.conv2d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv2d",
    ),
    skip(
        "nn.functional.cross_entropy",
        matcher=lambda sample: not isinstance(sample.kwargs.get("weight"), int),
        reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[weight] is int type",
    ),
    xfail(
        "nn.functional.embedding",
        matcher=lambda sample: sample.kwargs.get("max_norm") is not None,
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
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
    skip(
        "nn.functional.max_pool3d",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    ),
    xfail(
        "nonzero",
        matcher=lambda sample: len(sample.input.shape) == 0
        and sample.kwargs.get("as_tuple", False) is False,
        reason="Output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0]).",
        model_type=onnx_test_common.TorchModelType.TORCH_NN_MODULE,
    ),
    xfail(
        "nonzero",
        model_type=onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "aten::_assert_async.msg",
            "https://github.com/pytorch/pytorch/issues/112443",
        ),
    ),
    xfail(
        "scatter_add",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch",
    ),
    skip(
        "scatter_reduce",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    xfail(
        "unflatten",
        reason="Logic not implemented for size 0 inputs in op.Reshape",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
    ),
)

OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(meta.op_name for meta in SKIP_XFAIL_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)



class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


def _should_skip_xfail_test_sample(
    op_name: str, sample, model_type: onnx_test_common.TorchModelType
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
    for decorator_meta in SKIP_XFAIL_SUBTESTS:
        # Linear search on ops_test_data.SKIP_XFAIL_SUBTESTS. That's fine because the list is small.
        # NOTE: If model_type is None, the test is decorator_meta is meant to skip/xfail all model types.
        if decorator_meta.op_name == op_name and (
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
    rtol=1e-03,
    atol=1e-07,
):
    # avoid mutable default argument
    if input_kwargs is None:
        input_kwargs = {}

    # NOTE: ONNXProgram holds a reference (not copy) to the original ref_model, including its state_dict.
    # Thus, ONNXProgram() must run before ref_model() to prevent ref_model.forward() from changing the state_dict.
    # Otherwise, the ref_model can change buffers on state_dict which would be used by ONNXProgram.__call__()
    onnx_outputs = onnx_exported_program(*input_args, **input_kwargs)
    torch_outputs = torch_exported_program(*input_args, **input_kwargs)
    torch_outputs_onnx_format = onnx_exported_program.adapt_torch_outputs_to_onnx(
        torch_outputs
    )
    if len(torch_outputs_onnx_format) != len(onnx_outputs):
        raise AssertionError(
            f"Expected {len(torch_outputs_onnx_format)} outputs, got {len(onnx_outputs)}"
        )
    for torch_output, onnx_output in zip(torch_outputs_onnx_format, onnx_outputs):
        torch.testing.assert_close(
            torch_output, torch.tensor(onnx_output), rtol=rtol, atol=atol
        )

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
                op.name, cpu_sample, test_suite.model_type
            )
            with onnx_test_common.normal_xfail_skip_test_behaviors(
                test_behavior, reason
            ):
                model = SingleOpModel(op.op, cpu_sample.kwargs)
                model.eval()

                if dtype == torch.float32:
                    # Relax atol and rtol for float32 based on empirical results
                    rtol = 1e-5
                    atol = 2e-5
                elif (
                    dtype == torch.float16
                    and op.name in test_suite.fp16_low_precision_list
                ):
                    rtol = 2e-1
                    atol = 2e-2
                else:
                    rtol = None
                    atol = None

                if test_suite.model_type == onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM:
                    try:
                        model = torch.export.export(model, inputs)
                    except AssertionError as e:
                        # TODO: avoid fake_mode detection bug in torch.export.export
                        pytest.xfail(onnx_test_common.reason_dynamo_does_not_support(str(e)))
                
                try:
                    onnx_program = torch.onnx.dynamo_export(
                        model,
                        *inputs,
                    )
                except torch.onnx.OnnxExporterError as e:
                    # NOTE: If the model has unsupported nodes, we will skip the test
                    # with non-strict xfail. Otherwise, we will raise the error. 
                    if hasattr(e.__cause__, "diagnostic") and e.__cause__.diagnostic.rule in (
                        _rules._POERules.no_symbolic_function_for_call_function,
                        _rules._POERules.unsupported_fx_node_analysis
                    ):
                        pytest.xfail(onnx_test_common.reason_onnx_script_does_not_support(str(e)))
                    else:
                        raise e
                _compare_onnx_and_torch_exported_program(
                    model, onnx_program, inputs, rtol=rtol, atol=atol
                )


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (opset for opset in onnx_test_common.FX_TESTED_OPSETS),
            (
                onnx_test_common.TorchModelType.TORCH_NN_MODULE,
                onnx_test_common.TorchModelType.TORCH_EXPORT_EXPORTEDPROGRAM,
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
    model_type: onnx_test_common.TorchModelType = (
        onnx_test_common.TorchModelType.TORCH_NN_MODULE
    )

    # TODO: Make op have their own tolerance?
    fp16_low_precision_list = [
        "addbmm",
        "addcdic",
        "addcmul",
        "addmv",
        "addr",
        "baddbmm",
        "nn.functional.batch_norm",
        "native_batch_norm",
        "dot",
        "logit",
        "rsub",
        "sub",
    ]

    @common_device_type.ops(
        [op for op in OPS_DB if op.name in ALL_OPS_IN_DB],
        # TODO: Add back complex64
        allowed_dtypes=onnx_test_common.TESTED_DTYPES,
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        """Test the ONNX exporter."""
        _run_test_output_match(self, device, dtype, op)

for opset in onnx_test_common.FX_TESTED_OPSETS:
    for model_type in onnx_test_common.TorchModelType:
        # The name needs to match the parameterized_class name.
        test_class_name = f"TestOnnxModelOutputConsistency_opset_version_{opset}_model_type_TorchModelType.{model_type.name}"
        onnx_test_common.add_decorate_info(
            OPS_DB,
            test_class_name,
            "test_output_match",
            opset=opset,
            skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
        )
        
        common_device_type.instantiate_device_type_tests(
            globals()[test_class_name], globals(), only_for="cpu"
        )

if __name__ == "__main__":
    common_utils.run_tests()
