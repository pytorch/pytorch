# Owner(s): ["module: onnx"]

"""Test consistency between the output values of torch.onnx FX exported operators
and torch operators given the same inputs.

Usage:

    pytest test/onnx/test_op_consistency.py

    To run tests on a specific operator (e.g. torch.ceil):

    pytest test/onnx/test_op_consistency.py -k ceil
    pytest test/onnx/test_op_consistency.py -k nn_functional_scaled_dot_product_attention

    Read more on Running and writing tests:
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

Note:

    When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS and
    TESTED_OPS lists. See "Modify this section"

"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import onnx_test_common

import parameterized

import torch
from onnx_test_common import skip, xfail
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)

# Modify this section ##########################################################
# NOTE: Modify this section as more ops are supported. The list should be sorted
# alphabetically.
#
# For example, to add a test for torch.ceil:
# 1.  Add "ceil" to TESTED_OPS then run pytest.
# 2.  If the test fails, fix the error or add a new entry to EXPECTED_SKIPS_OR_FAILS.

# TODO: Directly modify DecorateInfo in each OpInfo in ob_db when all ops are enabled.
# Ops to be tested for numerical consistency between onnx and pytorch
TESTED_OPS: frozenset[str] = frozenset(
    [
        "abs",
        "acos",
        "acosh",
        "add",
        "addmm",
        "all",
        "allclose",
        "amax",
        "amin",
        "any",
        "arange",
        "argmax",
        "argmin",
        "as_strided",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "baddbmm",
        "bmm",
        "broadcast_to",
        "cat",
        "ceil",
        "chunk",
        "clamp",
        "clamp_max",
        "clamp_min",
        "clone",
        # "col2im", extra opinfo needed
        "constant_pad_nd",
        "contiguous",
        # "copy",  copy is not in OPS_DB
        "cos",
        "cosh",
        "cross",
        "cumsum",
        # "detach",  detach is not in OP-TEST-DB
        "div",
        "dot",
        # "empty",  non-deterministic
        # "empty_like",  non-deterministic
        # "empty_strided",  empty_strided is not in OPS_DB
        "eq",
        "equal",
        "erf",
        "exp",
        "exp2",
        "expand",
        "expand_as",
        "fill",
        "flip",
        "floor",
        "fmod",
        "full",
        "full_like",
        "gather",
        "hstack",  # aten::cat is invoked instead
        "index_put",
        "logit",
        "mean",
        # "new_empty",  non-deterministic
        # "new_empty_strided",  non-deterministic
        "new_full",
        "new_ones",
        "new_zeros",
        "nn.functional.adaptive_avg_pool1d",
        "nn.functional.adaptive_avg_pool2d",
        "nn.functional.adaptive_avg_pool3d",
        "nn.functional.conv1d",
        # "nn.functional.conv2d",  AssertionError: The values for attribute 'shape' do not match in float32
        # "nn.functional.conv3d",  extra opinfo needed
        # "nn.functional.convolution",  extra opinfo needed
        "nn.functional.cross_entropy",
        "nn.functional.celu",
        "nn.functional.dropout",
        "nn.functional.elu",
        "nn.functional.embedding",
        "nn.functional.max_pool1d",
        "nn.functional.max_pool2d",
        "nn.functional.max_pool3d",
        "nn.functional.nll_loss",
        # "nn.functional.scaled_dot_product_attention"  non-deterministic
        "scatter_add",
        "scatter_reduce",
        "square",
        "sum",
        "unflatten",
        # "var_mean",  # Segfault during onnx shape inference. Need to bump onnx version.
        "vstack",  # aten::cat is invoked instead
    ]
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
        "all",
        reason="[PostInline][ORT][ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0"
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
        "any",
        reason="[PostInline][ORT][ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0"
    ),
    xfail(
        "arange",
        dtypes=(torch.uint8, torch.int8),
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
        "cumsum", dtypes=onnx_test_common.BOOL_TYPES + (torch.uint8, torch.int8, torch.int16,),
        reason=onnx_test_common.reason_onnx_does_not_support("Cumsum", "bool, uint8, int8, int16")
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
        "nn.functional.adaptive_avg_pool1d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.index.Tensor"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool2d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.index.Tensor"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool3d",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.index.Tensor"),
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
        "nn.functional.embedding",
        reason=onnx_test_common.reason_onnx_script_does_not_support("aten.embedding_renorm.default"),
    ),
    xfail(
        "nn.functional.max_pool2d",
        dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Max_pool2d"),
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
        "unflatten", dtypes=onnx_test_common.BOOL_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Unflatten")
    ),
    xfail(
        "var_mean", dtypes=(torch.float16, ),
        reason=onnx_test_common.reason_onnx_script_does_not_support("var_mean", "float16")
    ),
    xfail(
        "var_mean", variant_name="unbiased", dtypes=(torch.float16, ),
        reason=onnx_test_common.reason_onnx_script_does_not_support("var_mean", "float16")
    ),
)
# fmt: on

SKIP_XFAIL_SUBTESTS: tuple[onnx_test_common.DecorateMeta, ...] = (
    xfail(
        "addmm",  # xfail can't only use dtypes to catch all cases
        matcher=lambda sample: sample.input.dtype
        in (torch.uint8, torch.int8, torch.int16),
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "Add", "int8, int16, uint8"
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
    skip(
        "cat",
        matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
        reason="core dump - cat does not support zero-dim tensors yet",
    ),
    skip(
        "div",
        matcher=lambda sample: sample.kwargs.get("rounding_mode") is not None,
        reason="rounding_mode is not yet supported",
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
    skip(
        "nn.functional.max_pool3d",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    ),
    skip(
        "nn.functional.nll_loss",
        matcher=lambda sample: isinstance(sample.kwargs.get("reduction"), str),
        reason=onnx_test_common.reason_onnx_script_does_not_support(
            "string in reduction kwarg: https://github.com/microsoft/onnxscript/issues/726"
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

# END OF SECTION TO MODIFY #####################################################


OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(meta.op_name for meta in SKIP_XFAIL_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)
# Assert all ops in OPINFO_FUNCTION_MAPPING are in the OPS_DB
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB"


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


def _should_skip_xfail_test_sample(
    op_name: str, sample
) -> Tuple[Optional[str], Optional[str]]:
    """Returns a reason if a test sample should be skipped."""
    if op_name not in OP_WITH_SKIPPED_XFAIL_SUBTESTS:
        return None, None
    for decorator_meta in SKIP_XFAIL_SUBTESTS:
        # Linear search on ops_test_data.SKIP_XFAIL_SUBTESTS. That's fine because the list is small.
        if decorator_meta.op_name == op_name:
            assert decorator_meta.matcher is not None, "Matcher must be defined"
            if decorator_meta.matcher(sample):
                return decorator_meta.test_behavior, decorator_meta.reason
    return None, None


def _get_test_class_name(cls, num, params_dict) -> str:
    del cls  # unused
    del num  # unused
    return params_dict["name"]


@parameterized.parameterized_class(
    [
        {
            "name": f"TestOnnxModelOutputConsistency_opset{opset}",
            "opset_version": opset,
        }
        for opset in onnx_test_common.FX_TESTED_OPSETS
    ],
    class_name_func=_get_test_class_name,
)
class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    opset_version = -1
    op_level_debug: bool = False
    dynamic_shapes: bool = False

    @common_device_type.ops(
        [op for op in OPS_DB if op.name in TESTED_OPS],
        allowed_dtypes=onnx_test_common.TESTED_DTYPES,
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        """Test the ONNX exporter."""
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

            with self.subTest(
                opset=self.opset_version,
                sample_num=i,
                inputs=repr(inputs),
                kwargs=repr(cpu_sample.kwargs),
            ):
                test_behavior, reason = _should_skip_xfail_test_sample(
                    op.name, cpu_sample
                )
                with onnx_test_common.normal_xfail_skip_test_behaviors(
                    test_behavior, reason
                ):
                    model = SingleOpModel(op.op, cpu_sample.kwargs)
                    model.eval()

                    if dtype == torch.float32:
                        # Relax atol and rtol for float32 based on empirical results
                        # The current most relaxed values are for aten::stft
                        rtol = 1e-5
                        atol = 2e-5
                    else:
                        rtol = None
                        atol = None
                    # Run the test
                    self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
                        model, inputs, rtol=rtol, atol=atol
                    )


for opset in onnx_test_common.FX_TESTED_OPSETS:
    # The name needs to match the parameterized_class name.
    test_class_name = f"TestOnnxModelOutputConsistency_opset{opset}"
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
