import random
import unittest

from itertools import product
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


def sample_inputs_window(op_info, device, dtype, requires_grad, *, kws=({},), **kwargs):
    _kwargs = dict(
        kwargs, **{"device": device, "dtype": dtype, "requires_grad": requires_grad}
    )
    sizes = [0, 1, 2, 5, 10, 50, 100, 1024, 2048]
    for size, periodic, k in product(sizes, (True, False), kws):
        yield SampleInput(size, periodic=periodic, **k, **_kwargs)


def sample_inputs_gaussian_window(op_info, device, dtype, requires_grad, **kwargs):
    kws = [{"std": random.uniform(0, 3)} for _ in range(50)]
    yield from sample_inputs_window(
        op_info, device, dtype, requires_grad, kws=kws, **kwargs
    )


def sample_inputs_exponential_window(op_info, device, dtype, requires_grad, **kwargs):
    kws = [{"center": None, "tau": random.uniform(0, 10)} for _ in range(50)]
    yield from sample_inputs_window(
        op_info, device, dtype, requires_grad, kws=kws, **kwargs
    )


def error_inputs_window(op_info, device, **kwargs):
    tmp_kwargs = dict(
        kwargs,
        **{
            "device": device,
        },
    )

    yield ErrorInput(
        SampleInput(-1, kwargs=tmp_kwargs),
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
        SampleInput(3, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="is not implemented for sparse types, got: torch.sparse_coo",
    )

    tmp_kwargs = kwargs
    tmp_kwargs.update(
        {
            "dtype": torch.long,
        }
    )

    yield ErrorInput(
        SampleInput(3, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="expects floating point dtypes, got: torch.int64",
    )


def error_inputs_exponential_window(op_info, device, **kwargs):
    tmp_kwargs = dict(kwargs, **{"dtype": torch.float32})

    yield from error_inputs_window(op_info, device, **tmp_kwargs)

    tmp_kwargs.update({"device": device})

    tmp_kwargs = dict(tmp_kwargs, **{"tau": -1})

    yield ErrorInput(
        SampleInput(3, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Tau must be positive, got: -1 instead.",
    )

    tmp_kwargs = dict(tmp_kwargs, **{"center": 1})

    yield ErrorInput(
        SampleInput(3, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Center must be 'None' for periodic equal True",
    )


def error_inputs_gaussian_window(op_info, device, **kwargs):
    tmp_kwargs = dict(kwargs, **{"dtype": torch.float32, "device": device})

    yield from error_inputs_window(op_info, device, **kwargs)

    tmp_kwargs = dict(tmp_kwargs, **{"std": -1})

    yield ErrorInput(
        SampleInput(3, kwargs=tmp_kwargs),
        error_type=ValueError,
        error_regex="Standard deviation must be positive, got: -1 instead.",
    )


def make_signal_windows_opinfo(
    name, variant_test_name, sample_inputs_func, error_inputs_func, *, skips=()
):
    return OpInfo(
        name=name,
        variant_test_name=variant_test_name,
        ref=scipy.signal.get_window if TEST_SCIPY else None,
        dtypes=all_types_and(torch.float, torch.double, torch.long),
        dtypesIfCUDA=all_types_and(
            torch.float, torch.double, torch.bfloat16, torch.half, torch.long
        ),
        sample_inputs_func=sample_inputs_func,
        error_inputs_func=error_inputs_func,
        supports_out=False,
        supports_autograd=False,
        skips=skips,
    )


op_db: List[OpInfo] = [
    make_signal_windows_opinfo(
        "signal.windows.cosine",
        "signal.windows.cosine_default",
        sample_inputs_window,
        error_inputs_window,
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
        "signal.windows.exponential",
        "signal.windows.exponential_default",
        sample_inputs_exponential_window,
        error_inputs_exponential_window,
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
        "signal.windows.gaussian",
        "signal.windows.gaussian_default",
        sample_inputs_gaussian_window,
        error_inputs_gaussian_window,
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
