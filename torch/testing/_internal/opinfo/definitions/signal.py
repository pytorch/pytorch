import random
import unittest
from functools import partial

from itertools import product
from typing import List

import torch
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    ErrorInput,
    OpInfo,
    SampleInput,
)
from torch.testing._legacy import floating_types_and

if TEST_SCIPY:
    import scipy.signal


def sample_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    r"""Base function used to create sample inputs for windows.

    For additional required args you should use *args, as well as **kwargs for
    additional keyword arguments.
    """

    # Test a window size of length zero and one.
    # If it's either symmetric or not doesn't matter in these sample inputs.
    for size in [0, 1]:
        yield SampleInput(
            size,
            *args,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )

    # For sizes larger than 1 we need to test both symmetric and non-symmetric windows.
    # Note: sample input tensors must be kept rather small.
    sizes = [2, 5, 10, 50]
    for size, sym in product(sizes, (True, False)):
        yield SampleInput(
            size,
            *args,
            sym=sym,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )


def sample_inputs_gaussian_window(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_window(
        op_info, device, dtype, requires_grad, random.uniform(0, 3), **kwargs  # std,
    )


def error_inputs_window(op_info, device, *args, **kwargs):
    # Tests for windows that have a negative size
    yield ErrorInput(
        SampleInput(-1, *args, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="requires non-negative window_length, got window_length=-1",
    )

    # Tests for window tensors that are not torch.strided, for instance, torch.sparse_coo.
    yield ErrorInput(
        SampleInput(
            3,
            *args,
            layout=torch.sparse_coo,
            device=device,
            dtype=torch.float32,
            **kwargs,
        ),
        error_type=ValueError,
        error_regex="is implemented for strided tensors only, got: torch.sparse_coo",
    )

    # Tests for window tensors that are not floating point dtypes, for instance, torch.long.
    yield ErrorInput(
        SampleInput(3, *args, dtype=torch.long, device=device, **kwargs),
        error_type=ValueError,
        error_regex="expects floating point dtypes, got: torch.int64",
    )


def error_inputs_exponential_window(op_info, device, **kwargs):
    # Yield common error inputs
    yield from error_inputs_window(op_info, device, 0.5, **kwargs)

    # Tests for negative decay values.
    yield ErrorInput(
        SampleInput(3, tau=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Tau must be positive, got: -1 instead.",
    )

    # Tests for non-symmetric windows and a given center value.
    yield ErrorInput(
        SampleInput(3, center=1, sym=False, dtype=torch.float32, device=device),
        error_type=ValueError,
        error_regex="Center must be 'None' for non-symmetric windows",
    )


def error_inputs_gaussian_window(op_info, device, **kwargs):
    # Yield common error inputs
    yield from error_inputs_window(op_info, device, 0.5, **kwargs)  # std

    # Tests for negative standard deviations
    yield ErrorInput(
        SampleInput(3, -1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Standard deviation must be positive, got: -1 instead.",
    )


def make_signal_windows_opinfo(
    name, variant_test_name, ref, sample_inputs_func, error_inputs_func, *, skips=()
):
    r"""Helper function to create OpInfo objects related to different windows."""
    return OpInfo(
        name=name,
        variant_test_name=variant_test_name,
        ref=ref if TEST_SCIPY else None,
        dtypes=floating_types_and(torch.bfloat16, torch.float16),
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=sample_inputs_func,
        error_inputs_func=error_inputs_func,
        supports_out=False,
        supports_autograd=False,
        skips=skips,
    )


op_db: List[OpInfo] = [
    make_signal_windows_opinfo(
        name="signal.windows.cosine",
        variant_test_name="signal.windows.cosine_default",
        ref=scipy.signal.windows.cosine,
        sample_inputs_func=sample_inputs_window,
        error_inputs_func=error_inputs_window,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # TODO: same as this?
            # https://github.com/pytorch/pytorch/issues/81774
            # also see: arange, new_full
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # skip these tests since we have non tensor input
            DecorateInfo(
                unittest.skip("Skipped!"), "TestCommon", "test_noncontiguous_samples"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_conj_view"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestMathBits", "test_neg_conj_view"
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_neg_view"),
            # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),
        ),
    ),
    make_signal_windows_opinfo(
        name="signal.windows.exponential",
        variant_test_name="signal.windows.exponential_default",
        ref=scipy.signal.windows.exponential,
        sample_inputs_func=partial(sample_inputs_window, tau=random.uniform(0, 10)),
        error_inputs_func=error_inputs_exponential_window,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # TODO: same as this?
            # https://github.com/pytorch/pytorch/issues/81774
            # also see: arange, new_full
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # skip these tests since we have non tensor input
            DecorateInfo(
                unittest.skip("Skipped!"), "TestCommon", "test_noncontiguous_samples"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_conj_view"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestMathBits", "test_neg_conj_view"
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_neg_view"),
            # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),
        ),
    ),
    make_signal_windows_opinfo(
        name="signal.windows.gaussian",
        variant_test_name="signal.windows.gaussian_default",
        ref=scipy.signal.windows.gaussian,
        sample_inputs_func=sample_inputs_gaussian_window,
        error_inputs_func=error_inputs_gaussian_window,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # TODO: same as this?
            # https://github.com/pytorch/pytorch/issues/81774
            # also see: arange, new_full
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # skip these tests since we have non tensor input
            DecorateInfo(
                unittest.skip("Skipped!"), "TestCommon", "test_noncontiguous_samples"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_conj_view"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestMathBits", "test_neg_conj_view"
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_neg_view"),
            # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),
        ),
    ),
]
