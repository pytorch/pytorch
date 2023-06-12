# Owner(s): ["module: onnx"]

"""Test consistency between the output values of torch.onnx exported operators
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

# For readability, these two are allowed to be imported as function
from onnx_test_common import skip, xfail
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)

OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

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
        "atan",
        "atan2",
        "broadcast_to",
        "ceil",
        "expand",
        "flatten",
        "logical_not",
        "nn.functional.scaled_dot_product_attention",
        "repeat",
        # "scatter_add",  # TODO: enable after fixing https://github.com/pytorch/pytorch/issues/102211
        # "scatter_reduce",  # TODO: enable after fixing https://github.com/pytorch/pytorch/issues/102211
        "sqrt",
        "stft",
        "t",
        "tile",
        "unflatten",
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
    skip(
        "atan", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Atan")
    ),
    xfail("atan", dtypes=[torch.float64], reason=onnx_test_common.reason_onnx_runtime_does_not_support("Atan", ["f64"])),
    skip(
        "atan2", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Atan")
    ),
    xfail("atan2", dtypes=[torch.float64], reason=onnx_test_common.reason_onnx_runtime_does_not_support("Atan", ["f64"])),
    xfail(
        "ceil", dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES,
        reason=onnx_test_common.reason_onnx_does_not_support("Ceil")
    ),
    skip("nn.functional.scaled_dot_product_attention", opsets=[onnx_test_common.opsets_before(14)], reason="Need Trilu."),
    skip("nn.functional.scaled_dot_product_attention", reason="fixme: ORT crashes on Windows, segfaults randomly on Linux"),
    skip("scatter_reduce", variant_name="amin", opsets=[onnx_test_common.opsets_before(16)],
         reason=onnx_test_common.reason_onnx_does_not_support("ScatterElements with reduction")),
    skip("scatter_reduce", variant_name="amax", opsets=[onnx_test_common.opsets_before(16)],
         reason=onnx_test_common.reason_onnx_does_not_support("ScatterElements with reduction")),
    skip("scatter_reduce", variant_name="prod", opsets=[onnx_test_common.opsets_before(16)],
         reason=onnx_test_common.reason_onnx_does_not_support("ScatterElements with reduction")),
    xfail("scatter_reduce", variant_name="mean",
          reason=onnx_test_common.reason_onnx_does_not_support("ScatterElements with reduction=mean")),
    skip("scatter_reduce", variant_name="sum", opsets=[onnx_test_common.opsets_before(16)],
         reason=onnx_test_common.reason_onnx_does_not_support("ScatterElements with reduction")),
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
    skip("sqrt", dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_does_not_support("Sqrt")),
    skip("stft", opsets=[onnx_test_common.opsets_before(17)], reason=onnx_test_common.reason_onnx_does_not_support("STFT")),
    xfail("stft",
          reason=onnx_test_common.reason_onnx_runtime_does_not_support("STFT", "Regression on ORT=1.15 4 percent difference")),
    skip("tile", opsets=[onnx_test_common.opsets_before(13)], reason=onnx_test_common.reason_onnx_does_not_support("Tile")),
    xfail("unflatten", opsets=[onnx_test_common.opsets_before(13)], reason="Helper function is needed to support legacy ops."),
)
# fmt: on

SKIP_XFAIL_SUBTESTS: tuple[onnx_test_common.DecorateMeta, ...] = (
    skip(
        "nn.functional.scaled_dot_product_attention",
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    ),
    skip(
        "repeat",
        reason="Empty repeats value leads to an invalid graph",
        matcher=lambda sample: not sample.args[0],
    ),
    skip(
        "scatter_reduce",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "stft",
        reason="ONNX STFT does not support complex results",
        matcher=lambda sample: sample.kwargs.get("return_complex") is True,
    ),
    skip(
        "tile",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape)
        or not sample.input.shape,
        reason="Logic not implemented for size 0 inputs in op.Reshape",
    ),
    skip(
        "unflatten",
        reason="Logic not implemented for size 0 inputs in op.Reshape",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
    ),
)


# END OF SECTION TO MODIFY #####################################################

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
        for opset in onnx_test_common.TESTED_OPSETS
    ],
    class_name_func=_get_test_class_name,
)
class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    opset_version = -1

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
                    model = SingleOpModel(op, cpu_sample.kwargs)
                    model.eval()

                    if dtype == torch.float32:
                        # Relax atol and rtol for float32 based on empirical results
                        # The current most relaxed values are for aten::stft
                        rtol = 1e-5
                        atol = 2e-5
                    elif dtype == torch.float64:
                        # The current most relaxed values are for aten::stft
                        rtol = 1e-5
                        atol = 2e-5
                    else:
                        rtol = None
                        atol = None
                    # Run the test
                    self.run_test(model, inputs, rtol=rtol, atol=atol)


for opset in onnx_test_common.TESTED_OPSETS:
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
