# Owner(s): ["module: onnx"]

"""Test consistency between torch.onnx exported operators and torch operators.

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
ORT_PROVIDERS = ("CPUExecutionProvider",)

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
    opsets: Collection[Union[int, Callable[[int], bool]]]
    device_type: Optional[str]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str

    def contains_opset(self, opset: int) -> bool:
        if not self.opsets:
            # Any empty container means all opsets
            return True
        return any(
            opset == opset_spec if isinstance(opset_spec, int) else opset_spec(opset)
            for opset_spec in self.opsets
        )


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    opsets: Collection[Union[int, Callable[[int], bool]]] = tuple(),
    device_type: Optional[str] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: str = "unspecified",
):
    """Expects a OpInfo test to fail."""
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        opsets=opsets,
        device_type=device_type,
        dtypes=dtypes,
        reason=reason,
    )


def dont_care(
    op_name: str,
    variant_name: str = "",
    *,
    opsets: Collection[Union[int, Callable[[int], bool]]] = tuple(),
    device_type: Optional[str] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason="unspecified",
):
    """Skips a test case in OpInfo that we don't care about.

    Likely because ONNX does not support the use case or it is by design.
    However, if ONNX changes its behavior and start to support the use case, we should
    update the test to expect the new behavior, leveraging XfailOpset.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Don't care: {reason}"),
        opsets=opsets,
        device_type=device_type,
        dtypes=dtypes,
        reason=reason,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    opsets: Collection[Union[int, Callable[[int], bool]]] = tuple(),
    device_type: Optional[str] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason="unspecified",
):
    """Skips a test case in OpInfo. It should be eventually fixed."""
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"To fix: {reason}"),
        opsets=opsets,
        device_type=device_type,
        dtypes=dtypes,
        reason=reason,
    )


def skip_ops(
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
            device_type=decorate_meta.device_type,
            dtypes=decorate_meta.dtypes,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def get_torch_op_name(op: Union[str, Callable]) -> str:
    """Returns the name of the torch function corresponding to the given op."""
    if callable(op):
        module_name = op.__module__.split("torch.", 1)
        op_name = op.__name__
        if len(module_name) == 2:
            # Remove the torch. prefix
            op_name = f"{module_name[1]}.{op_name}"
        return op_name
    # Already a string
    return op


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
# 1.  Add `torch.ceil` to ALLOWLIST_OP then run pytest.
#         You can also add a string, e.g. "ceil" or "__radd__".
# 2a. If the test fails, fix the error or add a new entry to EXPECTED_SKIPS_OR_FAILS.
# 2b. If the test is expected to fail only on certain opsets, add a new entry to
#     EXPECTED_OPSET_FAILS.

# TODO: Directly modify DecorateInfo in each OpInfo in ob_db when all ops are enabled.
# Ops to be tested for consistency between onnx and pytorch
ALLOWLIST_OP: AbstractSet[str] = frozenset(
    map(
        get_torch_op_name,
        (
            torch.ceil,
            torch.div,
            torch.floor_divide,
            torch.remainder,
            torch.sqrt,
            torch.t,
            torch.true_divide,
        ),
    )
)

# fmt: off
# Turn off black formatting to keep the list compact

# Expected failures for onnx export.
# The list should be sorted alphabetically by op name.
# Q: When should I use skip vs vs dont_care vs xfail?
# A: Use skip when we want to fix the test eventually but it doesn't fail consistently,
#        e.g. the test is flaky or some tests pass. Otherwise, use xfail.
#    Use dont_care if we don't care about the test passing, e.g. ONNX doesn't support the usage.
#    Use xfail if a test fails now and we want to eventually fix the test.
EXPECTED_SKIPS_OR_FAILS: Tuple[DecorateMeta, ...] = (
    dont_care(
        "ceil", dtypes=BOOL_TYPES + INT_TYPES + QINT_TYPES + COMPLEX_TYPES,
        reason=reason_onnx_does_not_support("Ceil")
    ),
    skip("ceil", dtypes=[torch.float64], reason=reason_onnx_runtime_does_not_support("Ceil", ["f64"])),
    xfail(
        "div", variant_name="no_rounding_mode", dtypes=COMPLEX_TYPES,
        reason=reason_jit_tracer_error("complex types")
    ),
    xfail(
        "div", variant_name="floor_rounding", dtypes=COMPLEX_TYPES,
        reason=reason_jit_tracer_error("complex types")
    ),
    xfail(
        "div", variant_name="trunc_rounding", dtypes=(torch.float16,) + COMPLEX_TYPES,
        reason=reason_jit_tracer_error("f16 and complex types")
    ),
    skip(
        "div", variant_name="no_rounding_mode", dtypes=[torch.uint8, torch.int8, torch.int16],
        reason=reason_onnx_runtime_does_not_support("Div", ["u8", "i8", "i16"])
    ),
    skip(
        "div", variant_name="floor_rounding", dtypes=[torch.uint8, torch.int8, torch.int16],
        reason=reason_onnx_runtime_does_not_support("Div", ["u8", "i8", "i16"])
    ),
    skip(
        "div", variant_name="floor_rounding", dtypes=[torch.float64],
        reason=reason_onnx_runtime_does_not_support("Div", ["f64"])
    ),
    skip(
        "div", variant_name="trunc_rounding", dtypes=[torch.uint8, torch.int8, torch.int16],
        reason=reason_onnx_runtime_does_not_support("Div", ["u8", "i8", "i16"])
    ),
    xfail("floor_divide", dtypes=COMPLEX_TYPES, reason=reason_jit_tracer_error("complex types")),
    skip("floor_divide", dtypes=[torch.float64], reason=reason_onnx_runtime_does_not_support("Floor", ["f64"])),
    xfail(
        "remainder", dtypes=[torch.uint8, torch.int8, torch.int16], opsets=[opsets_before(11)],
        reason="Sub not defined for u8, i16 before opset 14. Mod is used after 11 so we support from opset 11.",
    ),
    skip("remainder", dtypes=[torch.float64], reason=reason_onnx_runtime_does_not_support("Floor", ["f64"])),
    dont_care("sqrt", dtypes=BOOL_TYPES + QINT_TYPES + COMPLEX_TYPES, reason=reason_onnx_does_not_support("Sqrt")),
    xfail("t", dtypes=COMPLEX_TYPES, reason=reason_jit_tracer_error("complex types")),
    xfail("true_divide", dtypes=COMPLEX_TYPES, reason=reason_jit_tracer_error("complex types")),
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


class TestConsistency(common_utils.TestCase):
    """Test consistency of exported ONNX models.

    This is a parameterized test suite.
    """

    @classmethod
    def create_test_base(cls, opset: int):
        """Returns the base test method for the given opset."""

        def _output_match_base(self, device: str, dtype: torch.dtype, op):
            """Base test method for testing each opset."""
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

                    verification.verify(
                        model,
                        inputs,
                        input_kwargs={},
                        opset_version=opset,
                        keep_initializers_as_inputs=True,
                        ort_providers=ORT_PROVIDERS,
                        check_shape=True,
                        check_dtype=True,
                        flatten=True,
                    )

        return _output_match_base

    @classmethod
    def parameterize_opsets(cls, opsets: Sequence[int]):
        """Parameterizes the TestConsistency class with the given opsets."""
        for opset in opsets:
            # Generate a test method for each opset
            test_name = f"test_output_match_opset_{opset}"
            base_method = cls.create_test_base(opset)
            # Important to rename the test method so that DecorateInfo can find it
            base_method.__name__ = test_name

            # Update the ops to skip in the OpInfo database
            decorated = skip_ops(
                OPS_DB,
                cls.__name__,
                test_name,
                opset=opset,
                skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
            )(base_method)

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
            )(decorated)

            setattr(cls, test_name, decorated)


TestConsistency.parameterize_opsets(TESTED_OPSETS)
common_device_type.instantiate_device_type_tests(
    TestConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    common_utils.run_tests()
