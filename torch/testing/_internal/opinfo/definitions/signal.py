import unittest
from typing import List

import torch
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    ErrorInput,
    OpInfo,
    SampleInput,
)

if TEST_SCIPY:
    import scipy.signal


def _sample_input_windows(sample_input, *args, **kwargs):
    for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
        for periodic in [True, False]:
            kwargs.update(
                {
                    "periodic": periodic,
                }
            )
            yield sample_input(size, args=args, kwargs=kwargs)


def sample_inputs_window(op_info, *args, **kwargs):
    return _sample_input_windows(SampleInput, *args, **kwargs)


def error_inputs_window(op_info, *args, **kwargs):
    yield ErrorInput(
        SampleInput(-1, args=args, kwargs=kwargs),
        error_type=ValueError,
        error_regex="requires non-negative window_length, got window_length=-1",
    )

    tmp_kwargs = dict(
        kwargs,
        **{
            "layout": torch.sparse_coo,
        },
    )

    yield ErrorInput(
        SampleInput(3, args=args, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="is not implemented for sparse types, got: torch.sparse_coo",
    )

    tmp_kwargs = kwargs
    tmp_kwargs["dtype"] = torch.long

    yield ErrorInput(
        SampleInput(3, args=args, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="expects floating point dtypes, got: torch.int64",
    )


def error_inputs_exponential_window(op_info, device, *args, **kwargs):
    tmp_kwargs = dict(kwargs, **{"dtype": torch.float32, "device": device})

    for error_input in error_inputs_window(op_info, *args, **kwargs):
        yield error_input

    tmp_kwargs = dict(tmp_kwargs, **{"tau": -1})

    yield ErrorInput(
        SampleInput(3, args=args, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Tau must be positive, got: -1 instead.",
    )

    tmp_kwargs = dict(tmp_kwargs, **{"center": 1})

    yield ErrorInput(
        SampleInput(3, args=args, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Center must be 'None' for periodic equal True",
    )


def error_inputs_gaussian_window(op_info, device, *args, **kwargs):
    tmp_kwargs = dict(kwargs, **{"dtype": torch.float32, "device": device})

    for error_input in error_inputs_window(op_info, *args, **kwargs):
        yield error_input

    tmp_kwargs = dict(tmp_kwargs, **{"std": -1})

    yield ErrorInput(
        SampleInput(3, args=args, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Standard deviation must be positive, got: -1 instead.",
    )


op_db: List[OpInfo] = [
    OpInfo(
        "signal.windows.cosine",
        ref=scipy.signal.get_window if TEST_SCIPY else None,
        dtypes=all_types_and(torch.float, torch.double, torch.long),
        dtypesIfCUDA=all_types_and(
            torch.float, torch.double, torch.bfloat16, torch.half, torch.long
        ),
        sample_inputs_func=sample_inputs_window,
        error_inputs_func=error_inputs_window,
        supports_out=False,
        supports_autograd=False,
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
    OpInfo(
        "signal.windows.exponential",
        ref=scipy.signal.get_window if TEST_SCIPY else None,
        dtypes=all_types_and(torch.float, torch.double, torch.long),
        dtypesIfCUDA=all_types_and(
            torch.float, torch.double, torch.bfloat16, torch.half, torch.long
        ),
        sample_inputs_func=sample_inputs_window,
        error_inputs_func=error_inputs_exponential_window,
        supports_out=False,
        supports_autograd=False,
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
    OpInfo(
        "signal.windows.gaussian",
        ref=scipy.signal.get_window if TEST_SCIPY else None,
        dtypes=all_types_and(torch.float, torch.double, torch.long),
        dtypesIfCUDA=all_types_and(
            torch.float, torch.double, torch.bfloat16, torch.half, torch.long
        ),
        sample_inputs_func=sample_inputs_window,
        error_inputs_func=error_inputs_gaussian_window,
        supports_out=False,
        supports_autograd=False,
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

python_ref_db: List[OpInfo] = []
