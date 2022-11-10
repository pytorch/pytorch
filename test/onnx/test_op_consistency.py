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
    ALLOWLIST_OP lists. See "Modify this section"

"""

import copy
import dataclasses
import io
import unittest
from typing import (
    AbstractSet,
    Callable,
    Collection,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import onnx
import onnx_test_common

import torch
from torch.onnx import _constants, verification
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

SUPPORTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # Floating types
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    # QInt types
    torch.qint8,
    torch.quint8,
    # Complex types
    torch.complex32,
    torch.complex64,
    torch.complex128,
)

# Convenience tuples for creating dtype lists when skipping or xfailing tests

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
    torch.bfloat16,
)

COMPLEX_TYPES = (
    torch.complex32,
    torch.complex64,
    torch.complex128,
)


@dataclasses.dataclass
class DecorateMeta:
    """A dataclass for storing information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py
    """

    op_name: str
    variant_name: str
    decorator: Callable
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str

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
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: Optional[str] = None,
):
    """Expects a OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    if reason is None:
        raise ValueError("Please specify a reason.")
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
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: Optional[str] = None,
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    if reason is None:
        raise ValueError("Please specify a reason.")
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Don't care: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
    )


def fixme(
    op_name: str,
    variant_name: str = "",
    *,
    opsets: Optional[Collection[Union[int, Callable[[int], bool]]]] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: Optional[str] = None,
):
    """Skips a test case in OpInfo. It should be eventually fixed.

    Args:
        op_name: The name of the operator.
        variant_name: The name of the variant.
        opsets: The opsets to expect the failure. e.g. [9, 10] or [opsets_before(11)]
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    if reason is None:
        raise ValueError("Please specify a reason.")
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"To fix: {reason}"),
        opsets=opsets,
        dtypes=dtypes,
        reason=reason,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    opset: int,
    skip_or_xfails: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list."""
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
# 1.  Add "ceil" to ALLOWLIST_OP then run pytest.
# 2.  If the test fails, fix the error or add a new entry to EXPECTED_SKIPS_OR_FAILS.

# TODO: Directly modify DecorateInfo in each OpInfo in ob_db when all ops are enabled.
# Ops to be tested for numerical consistency between onnx and pytorch
ALLOWLIST_OP: AbstractSet[str] = frozenset(
    [
        "ceil",
        "sqrt",
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
        "ceil", dtypes=BOOL_TYPES + INT_TYPES + QINT_TYPES + COMPLEX_TYPES,
        reason=reason_onnx_does_not_support("Ceil")
    ),
    fixme("ceil", dtypes=[torch.float64], reason=reason_onnx_runtime_does_not_support("Ceil", ["f64"])),
    dont_care("sqrt", dtypes=BOOL_TYPES + QINT_TYPES + COMPLEX_TYPES, reason=reason_onnx_does_not_support("Sqrt")),
    xfail("t", dtypes=COMPLEX_TYPES, reason=reason_jit_tracer_error("complex types")),
)
# fmt: on


# END OF SECTION TO MODIFY #####################################################


OPS_DB = copy.deepcopy(common_methods_invocations.op_db)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    @classmethod
    def create_test_base(cls, opset: int):
        """Returns the base test method for the given opset."""

        def _output_match_base(self, device: str, dtype: torch.dtype, op):
            """Base test method for testing each opset, used by instantiate_device_type_tests."""
            # device is provided by instantiate_device_type_tests, but we only want to run in cpu.
            assert device == "cpu"

            samples = op.sample_inputs(
                device,
                dtype,
                requires_grad=False,
            )

            for (i, cpu_sample) in enumerate(samples):
                # Provide the repr to subtest because tensors are not serializable in parallel test runs
                with self.subTest(
                    opset=opset,
                    sample_num=i,
                    input=repr(cpu_sample.input),
                    args=repr(cpu_sample.args),
                    kwargs=repr(cpu_sample.kwargs),
                ):
                    model = SingleOpModel(op, cpu_sample.kwargs)
                    model.eval()

                    # Run the test
                    inputs = (cpu_sample.input, *cpu_sample.args)

                    if dtype == torch.bfloat16:
                        # Only export to ONNX without running with onnxruntime because
                        # the CPU execution path for bfloat16 is not implemented in onnxruntime.
                        model_buffer = io.BytesIO()
                        torch.onnx.export(
                            model, inputs, model_buffer, opset_version=opset
                        )
                        model_buffer.seek(0)
                        onnx_model = onnx.load(model_buffer)
                        onnx.checker.check_model(onnx_model, full_check=True)
                        continue

                    self.run_test(model, inputs)

        test_name = f"test_output_match_opset_{opset}"
        _output_match_base.__name__ = test_name
        return _output_match_base

    @classmethod
    def parameterize_opsets(cls, opsets: Sequence[int]):
        """Parameterizes the TestOnnxModelOutputConsistency class with the given opsets."""
        for opset in opsets:
            # Generate a test method for each opset
            base_method = cls.create_test_base(opset)
            # Important to rename the test method so that DecorateInfo can find it
            test_name = base_method.__name__

            # Update the ops to skip in the OpInfo database
            add_decorate_info(
                OPS_DB,
                cls.__name__,
                test_name,
                opset=opset,
                skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
            )

            # Create parameterized tests for each op
            if opset < 13:
                # bfloat16 is not supported before opset 13
                allowed_dtypes = tuple(
                    [dtype for dtype in SUPPORTED_DTYPES if dtype != torch.bfloat16]
                )
            else:
                allowed_dtypes = SUPPORTED_DTYPES
            filtered_ops = [op for op in OPS_DB if op.name in ALLOWLIST_OP]
            decorated = common_device_type.ops(
                filtered_ops,
                allowed_dtypes=allowed_dtypes,
            )(base_method)

            setattr(cls, test_name, decorated)


TestOnnxModelOutputConsistency.parameterize_opsets(TESTED_OPSETS)
common_device_type.instantiate_device_type_tests(
    TestOnnxModelOutputConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    common_utils.run_tests()
