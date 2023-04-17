# Owner(s): ["module: onnx"]

"""Test consistency between the output values of torch.onnx exported operators
and torch operators given the same inputs.

Usage:

    pytest test/onnx/test_op_consistancy.py

    To run tests on a specific operator (e.g. torch.ceil):

    pytest test/onnx/test_op_consistancy.py -k ceil

    Read more on Running and writing tests:
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

Note:

    When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS and
    TESTED_OPS lists. See "Modify this section"

"""

from __future__ import annotations

import copy
import dataclasses
import unittest
import warnings
from typing import Any, Callable, Collection, Iterable, Optional, Sequence, Tuple, Union

import onnx_test_common
import parameterized

import torch
from torch.onnx import _constants
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.opinfo import core as opinfo_core

# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_MAX_OPSET

TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)

QINT_TYPES = (
    torch.qint8,
    torch.quint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
)

COMPLEX_TYPES = (
    torch.complex32,
    torch.complex64,
    torch.complex128,
)

TESTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    *INT_TYPES,
    # Floating types
    *FLOAT_TYPES,
)


@dataclasses.dataclass
class DecorateMeta:
    """Information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py

    Attributes:
        op_name: The name of the operator.
        variant_name: The name of the OpInfo variant.
        decorator: The decorator to apply to the test case.
        opsets: The opsets to apply the decorator to.
        dtypes: The dtypes to apply the decorator to.
        reason: The reason for skipping.
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str
    matcher: Optional[Callable[[Any], Any]] = None

    def contains_opset(self, opset: int) -> bool:
        if self.opsets is None:
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
):
    """Expects a OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
    )


def dont_care(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            dont_care is in the SKIP_SUBTESTS list.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Don't care: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
    )


def fixme(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
):
    """Skips a test case in OpInfo. It should be eventually fixed.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
        matcher: A function that matches the test sample input. It is used only when
            fixme is in the SKIP_SUBTESTS list.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"To fix: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list.

    Args:
        all_opinfos: All OpInfos.
        test_class_name: The name of the test class.
        base_test_name: The name of the test method.
        opset: The opset to decorate for.
        skip_or_xfails: DecorateMeta's.
    """
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        if not decorate_meta.contains_opset(opset):
            # Skip does not apply to this opset
            continue
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        assert (
            opinfo is not None
        ), f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def opsets_before(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is before the specified."""

    def compare(other_opset: int):
        return other_opset < opset

    return compare


def opsets_after(opset: int) -> Callable[[int], bool]:
    """Returns a comparison function that decides if the given opset is after the specified."""

    def compare(other_opset: int):
        return other_opset > opset

    return compare


def reason_onnx_runtime_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX Runtime doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'dtypes'} not supported by ONNX Runtime"


def reason_onnx_does_not_support(
    operator: str, dtypes: Optional[Sequence[str]] = None
) -> str:
    """Formats the reason: ONNX doesn't support the given dtypes."""
    return f"{operator} on {dtypes or 'certain dtypes'} not supported by the ONNX Spec"


def reason_jit_tracer_error(info: str) -> str:
    """Formats the reason: JIT tracer errors."""
    return f"JIT tracer error on {info}"


def reason_flaky() -> str:
    """Formats the reason: test is flaky."""
    return "flaky test"


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
        "ceil",
        "logical_not",
        "sqrt",
        "stft",
        "t",
    ]
)

# fmt: off
# Turn off black formatting to keep the list compact

# Expected failures for onnx export.
# The list should be sorted alphabetically by op name.
# Q: When should I use fixme vs vs dont_care vs xfail?
# A: Use fixme when we want to fix the test eventually but it doesn't fail consistently,
#        e.g. the test is flaky or some tests pass. Otherwise, use xfail.
#    Use dont_care if we don't care about the test passing, e.g. ONNX doesn't support the usage.
#    Use xfail if a test fails now and we want to eventually fix the test.
EXPECTED_SKIPS_OR_FAILS: Tuple[DecorateMeta, ...] = (
    dont_care(
        "ceil", dtypes=BOOL_TYPES + INT_TYPES,
        reason=reason_onnx_does_not_support("Ceil")
    ),
    fixme("ceil", dtypes=[torch.float64], reason=reason_onnx_runtime_does_not_support("Ceil", ["f64"])),
    dont_care("sqrt", dtypes=BOOL_TYPES, reason=reason_onnx_does_not_support("Sqrt")),
    dont_care("stft", opsets=[opsets_before(17)], reason=reason_onnx_does_not_support("STFT")),
)
# fmt: on

SKIP_SUBTESTS: tuple[DecorateMeta, ...] = (
    dont_care(
        "stft",
        reason="ONNX STFT does not support complex results",
        matcher=lambda sample: sample.kwargs.get("return_complex") is True,
    ),
)

# END OF SECTION TO MODIFY #####################################################


OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_SUBTESTS = frozenset(meta.op_name for meta in SKIP_SUBTESTS)
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


def _should_skip_test_sample(op_name: str, sample) -> Optional[str]:
    """Returns a reason if a test sample should be skipped."""
    if op_name not in OP_WITH_SKIPPED_SUBTESTS:
        return None
    for decorator_meta in SKIP_SUBTESTS:
        # Linear search on SKIP_SUBTESTS. That's fine because the list is small.
        if decorator_meta.op_name == op_name:
            assert decorator_meta.matcher is not None, "Matcher must be defined"
            if decorator_meta.matcher(sample):
                return decorator_meta.reason
    return None


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
        for opset in TESTED_OPSETS
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
        allowed_dtypes=TESTED_DTYPES,
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
                skip_reason = _should_skip_test_sample(op.name, cpu_sample)
                if skip_reason is not None:
                    # Cannot use self.skip because pytest would skip the entire test
                    warnings.warn(f"skipped sample {i}. Reason: {skip_reason}")
                    continue

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


for opset in TESTED_OPSETS:
    # The name needs to match the parameterized_class name.
    test_class_name = f"TestOnnxModelOutputConsistency_opset{opset}"
    add_decorate_info(
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
