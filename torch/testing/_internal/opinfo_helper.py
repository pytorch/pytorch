import collections
import warnings
from functools import partial

import torch
from torch.testing._internal.common_cuda import (TEST_CUDA)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    all_types_and_complex,
    all_types_and_half,
    all_types,
    complex_types,
    floating_and_complex_types,
    floating_types_and_half,
    floating_types,
    integral_types,
    floating_types_and,
    floating_and_complex_types_and,
    integral_types_and,
    all_types_and,
    _dispatch_dtypes,
)

COMPLETE_DTYPES_DISPATCH = (
    all_types,
    all_types_and_complex,
    all_types_and_half,
    floating_types,
    floating_and_complex_types,
    floating_types_and_half,
    integral_types,
    complex_types,
)

EXTENSIBLE_DTYPE_DISPATCH = (
    all_types_and_complex_and,
    floating_types_and,
    floating_and_complex_types_and,
    integral_types_and,
    all_types_and,
)

# Better way to acquire devices?
DEVICES = ['cpu'] + (['cuda'] if TEST_CUDA else [])

class _dynamic_dispatch_dtypes(_dispatch_dtypes):
    # Class to tag the dynamically generated types.
    pass


def get_supported_dtypes(op, sample_inputs_fn, device_type):
    # Returns the supported dtypes for the given operator and device_type pair.
    assert device_type in ['cpu', 'cuda']
    if not TEST_CUDA and device_type == 'cuda':
        warnings.warn("WARNING: CUDA is not available, empty_dtypes dispatch will be returned!")
        return _dynamic_dispatch_dtypes(())

    supported_dtypes = set()
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        try:
            samples = sample_inputs_fn(op, device_type, dtype, False)
        except RuntimeError:
            # If `sample_inputs_fn` doesn't support sampling for a given
            # `dtype`, we assume that the `dtype` is not supported.
            # We raise a warning, so that user knows that this was the case
            # and can investigate if there was an issue with the `sample_inputs_fn`.
            warnings.warn(f"WARNING: Unable to generate sample for device:{device_type} and dtype:{dtype}")
            continue

        # We assume the dtype is supported
        # only if all samples pass for the given dtype.
        supported = True
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)
            except RuntimeError as re:
                # dtype is not supported
                supported = False
                break

        if supported:
            supported_dtypes.add(dtype)

    return _dynamic_dispatch_dtypes(supported_dtypes)


def dtypes_dispatch_hint(dtypes):
    # Function returns the appropriate dispatch function (from COMPLETE_DTYPES_DISPATCH and EXTENSIBLE_DTYPE_DISPATCH)
    # and its string representation for the passed `dtypes`.
    return_type = collections.namedtuple('return_type', 'dispatch_fn dispatch_fn_str')

    # CUDA is not available, dtypes will be empty.
    if len(dtypes) == 0:
        return return_type((), str(tuple()))

    set_dtypes = set(dtypes)
    for dispatch in COMPLETE_DTYPES_DISPATCH:
        # Short circuit if we get an exact match.
        if set(dispatch()) == set_dtypes:
            return return_type(dispatch, dispatch.__name__ + "()")

    chosen_dispatch = None
    chosen_dispatch_score = 0.
    for dispatch in EXTENSIBLE_DTYPE_DISPATCH:
        dispatch_dtypes = set(dispatch())
        if not dispatch_dtypes.issubset(set_dtypes):
            continue

        score = len(dispatch_dtypes)
        if score > chosen_dispatch_score:
            chosen_dispatch_score = score
            chosen_dispatch = dispatch

    # If user passed dtypes which are lower than the lowest
    # dispatch type available (not likely but possible in code path).
    if chosen_dispatch is None:
        return return_type((), str(dtypes))

    return return_type(partial(dispatch, *tuple(set(dtypes) - set(dispatch()))),
                       dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))))


def is_dynamic_dtype_set(op):
    # Detect if the OpInfo entry acquired dtypes dynamically
    # using `get_supported_dtypes`.
    return op.dynamic_dtypes


def str_format_dynamic_dtype(op):
    fmt_str = """
        OpInfo({name},
               dtypes={dtypesIfCPU},
               dtypesIfCUDA={dtypesIfCUDA},
        )
        """.format(name=op.name,
                   dtypesIfCPU=dtypes_dispatch_hint(op.dtypesIfCPU).dispatch_fn_str,
                   dtypesIfCUDA=dtypes_dispatch_hint(op.dtypesIfCUDA).dispatch_fn_str)

    return fmt_str
