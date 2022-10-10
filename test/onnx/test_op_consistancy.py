# Owner(s): ["module: onnx"]

"""Test consistency between torch.onnx exported operators and aten operators.

NOTE:

When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS and
ALLOWLIST_OP lists.

"""
import itertools
import unittest
from collections import namedtuple
from typing import Collection, Iterable, Optional, Sequence

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


# Copied from functorch
# A named tuple for storing information about a test case to skip
DecorateMeta = namedtuple(
    "DecorateMeta",
    [
        "op_name",
        "variant_name",
        "decorator",
        "device_type",
        "dtypes",
        "reason",
    ],
)


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    device_type: Optional[str] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: str = "unspecified",
):
    """Expects a OpInfo test to fail."""
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        device_type=device_type,
        dtypes=dtypes,
        reason=reason,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    device_type: Optional[str] = None,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason="unspecified",
):
    """Skips a test case in OpInfo."""
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(reason),
        device_type=device_type,
        dtypes=dtypes,
        reason=reason,
    )


def skip_ops(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_case_name: str,
    base_test_name: str,
    to_skip: Iterable[DecorateMeta],
):
    """Decorates OpInfo tests with decorators based on the to_skip list."""
    for decorate_meta in to_skip:
        matching_opinfos = [
            info
            for info in all_opinfos
            if info.name == decorate_meta.op_name
            and info.variant_test_name == decorate_meta.variant_name
        ]
        assert len(matching_opinfos) > 0, f"Couldn't find OpInfo for {decorate_meta}"
        assert len(matching_opinfos) == 1, (
            "OpInfos should be uniquely determined by their (name, variant_name). "
            f"Got more than one result for ({decorate_meta.op_name}, {decorate_meta.variant_name})"
        )
        opinfo = matching_opinfos[0]
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            test_case_name,
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


# NOTE: Modify this section as more ops are supported. The list should be sorted
# alphabetically.

EXPECTED_SKIPS_OR_FAILS = (
    xfail(
        "ceil",
        dtypes=(torch.bfloat16, torch.float64),
        reason="Ceil not implemented for f64 and bf64 in onnx runtime",
    ),
    skip(
        "ceil",
        dtypes=BOOL_TYPES + INT_TYPES + QINT_TYPES + COMPLEX_TYPES,
        reason="Not supported by onnx",
    ),
    xfail(
        "sqrt",
        dtypes=(torch.bfloat16,),
        reason="Sqrt not implemented for bf64 in onnx runtime",
    ),
    skip(
        "sqrt",
        dtypes=BOOL_TYPES + INT_TYPES + QINT_TYPES + COMPLEX_TYPES,
        reason="Not supported by onnx",
    ),
    xfail(
        "t",
        dtypes=(torch.bfloat16,) + COMPLEX_TYPES,
        reason="Transpose not implemented for bf64 in onnx runtime",
    ),
)


# Ops and the dtypes to be tested for consistency
# NOTES: Incrementally add / uncomment ops and add types to this list as they are supported
# The list should be kept reasonably sorted

ALLOWLIST_OP = (
    "ceil",
    "sqrt",
    "t",
)


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
    @common_device_type.ops(
        common_methods_invocations.op_db, allowed_dtypes=SUPPORTED_DTYPES
    )
    @skip_ops(
        common_methods_invocations.op_db,
        "TestConsistency",
        "test_output_match",
        to_skip=EXPECTED_SKIPS_OR_FAILS,
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        assert device == "cpu"

        if op.name not in ALLOWLIST_OP:
            self.skipTest(f"'{op.name}' is not in the allow list for test on ONNX")

        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=False,
        )

        for (cpu_sample, opset) in itertools.product(samples, TESTED_OPSETS):
            model = SingleOpModel(op, cpu_sample.kwargs)

            with self.subTest(sample=cpu_sample, opset=opset):
                verification.verify(
                    model,
                    (cpu_sample.input, *cpu_sample.args),
                    input_kwargs={},
                    opset_version=opset,
                    keep_initializers_as_inputs=True,
                    ort_providers=ORT_PROVIDERS,
                    check_shape=True,
                    check_dtype=True,
                    flatten=True,
                )


common_device_type.instantiate_device_type_tests(
    TestConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    common_utils.run_tests()
