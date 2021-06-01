import collections
import warnings
from functools import partial

import torch
from torch.testing._internal.common_cuda import (TEST_CUDA)
from torch.testing._core import _dispatch_dtypes
from torch.testing import (all_types_and_complex_and,
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

# Class to tag the dynamically generated types.


class _dynamic_dispatch_dtypes(_dispatch_dtypes):
    pass


def get_supported_dtypes(op, sample_inputs_fn, device_type):
    # Returns the supported dtypes for the given operator and device_type pair.
    assert device_type in ['cpu', 'cuda']
    if not TEST_CUDA and device_type == 'cuda':
        warnings.warn("WARNING: CUDA is not available, information pertaining to CUDA dtypes could be wrong")
        return _dynamic_dispatch_dtypes(())

    supported_dtypes = set()
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        samples = sample_inputs_fn(op, device_type, dtype, False)
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)
                supported_dtypes.add(dtype)
            except RuntimeError as re:
                if "not implemented for" not in str(re):
                    raise re
                pass

    return _dynamic_dispatch_dtypes(supported_dtypes)


def dtypes_dispatch_hint(dtypes):
    return_type = collections.namedtuple('return_type', 'dispatch_fn dispatch_fn_str')

    # CUDA is not available, dtypes will be empty.
    if len(dtypes) == 0:
        return return_type((), str(tuple()))

    # Find the closest dispatcher from the set of available dispatchers

    # Metric for finding the closest dispatcher
    # Intersection Over Union
    def iou(dispatch_dtypes):
        return len(set(dtypes).intersection(dispatch_dtypes)) * 1.0 / len(set(dtypes).union(dispatch_dtypes))

    dispatch_iou_score = {dispatch: iou(dispatch()) for dispatch in COMPLETE_DTYPES_DISPATCH}

    dispatch, score = max(dispatch_iou_score.items(), key=lambda x: x[1])

    if dispatch in COMPLETE_DTYPES_DISPATCH and score == 1.0:
        return return_type(dispatch, dispatch.__name__ + "()")

    def subset_or_equal(dispatch_dtypes):
        return len(set(dispatch_dtypes) - set(dtypes)) == 0

    filtered_dispatches = list(filter(lambda dispatch: subset_or_equal(dispatch()), EXTENSIBLE_DTYPE_DISPATCH))

    # If user passed dtypes which are lower than the lowest
    # dispatch type available (not likely but possible in code path).
    if len(filtered_dispatches) == 0:
        return return_type((), str(dtypes))

    dispatch_iou_score = {dispatch: iou(dispatch()) for dispatch in filtered_dispatches}
    dispatch, _ = max(dispatch_iou_score.items(), key=lambda x: x[1])

    return return_type(partial(dispatch, *tuple(set(dtypes) - set(dispatch()))),
                       dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))))


def is_dynamic_dtype_set(op):
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


# Run using `python -m torch.testing._internal.opinfo_helper`
if __name__ == '__main__':
    # import here to break circular dependency(?)
    import torch.testing._internal.common_methods_invocations as cmi

    filtered_ops = list(filter(is_dynamic_dtype_set, cmi.op_db))
    print("The operator/s below is using dynamic_dtypes in the OpInfo entry!")
    for op in filtered_ops:
        print(str_format_dynamic_dtype(op))
