from collections import defaultdict
import warnings
from functools import partial

import torch
from torch.testing._internal.common_cuda import (TEST_CUDA)
from torch.testing._internal.common_methods_invocations import (op_db, _DYNAMIC_DTYPES)
from torch.testing import (all_types_and_complex_and,
                           all_types_and_complex,
                           # all_types_and_half, # not exported by testing at the moment
                           all_types,
                           complex_types,
                           floating_and_complex_types,
                           # floating_types_and_half, # not exported by testing at the moment
                           floating_types,
                           integral_types,
                           )

DTYPE_DISPATCH = (
    floating_types,
    floating_and_complex_types,
    integral_types,
    all_types,
    complex_types,
    all_types_and_complex,
)

# Better way to acquire devices?
DEVICES = ['cpu'] + (['cuda'] if TEST_CUDA else [])

if not TEST_CUDA:
    warnings.warn("WARNING: CUDA is not available, information pertaining to CUDA could be wrong")


class OpInfoHelper:
    def __init__(self, op_info):
        self.op = op_info
        self.sample_inputs_fn = op_info.sample_inputs_func

    def get_supported_dtypes(self):
        # Return a map of (device, supported_dtypes)
        device_dtype = defaultdict(list)
        for device in DEVICES:
            for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
                sample = self.sample_inputs_fn(self.op, device, dtype, False)[0]
                try:
                    self.op(sample.input, *sample.args, **sample.kwargs)
                    device_dtype[device].append(dtype)
                except RuntimeError as re:
                    if "not implemented for" not in str(re):
                        raise re
                    pass

        return device_dtype

    def has_out_support(self, device, dtype):
        try:
            sample = self.sample_inputs_fn(None, device, dtype, False)[0]
            op_out = partial(self.op, sample.input, *sample.args, **sample.kwargs)
            op_out(out=torch.empty_like(sample.input))
            return True
        except TypeError as te:
            if "unexpected keyword argument 'out'" not in str(te):
                raise te
            return False

    def dtype_dispatch_hint(self, dtypes):
        # Find the closest dispatcher from the set of available dispatchers

        # Metric for finding the closest dispatcher
        # Intersection Over Union
        def iou(dispatch_dtypes):
            return len(set(dtypes).intersection(dispatch_dtypes)) * 1.0 / len(set(dtypes).union(dispatch_dtypes))

        dispatch_iou_score = {dispatch: iou(dispatch()) for dispatch in DTYPE_DISPATCH}

        dispatch, _ = max(dispatch_iou_score.items(), key=lambda x: x[1])

        return dispatch.__name__ + " + " + str(tuple(set(dtypes) - set(dispatch())))

    def summary(self):
        device_supported_dtypes = self.get_supported_dtypes()
        supports_out = self.has_out_support('cpu', device_supported_dtypes['cpu'][0])

        # `dtypes` argument of OpInfo
        cpu_dtypes = device_supported_dtypes["cpu"]
        cpu_dispatch_hint = self.dtype_dispatch_hint(cpu_dtypes)

        # `dtypesIfCUDA` argument of OpInfo
        cuda_dtypes = device_supported_dtypes['cuda']
        cuda_dispatch_hint = self.dtype_dispatch_hint(cuda_dtypes)

        summary_str = (f'OpInfo({self.op.name}',
                       f'# hint: {cpu_dispatch_hint}',
                       f'dtypes={cpu_dtypes}',
                       f'# hint: {cuda_dispatch_hint}' if (cpu_dtypes != cuda_dtypes and TEST_CUDA) else '',
                       f'dtypesIfCUDA={cuda_dtypes}' if (cpu_dtypes != cuda_dtypes and TEST_CUDA) else '',
                       f'supports_out={supports_out}' if not supports_out else '',  # as supports_out is True by default
                       f'sample_inputs_func={self.sample_inputs_fn.__name__})')

        empty_spaces = " " * len('OpInfo(')
        return (",\n" + empty_spaces).join(filter(None, summary_str))


# Run using `python -m torch.testing._internal.opinfo_helper`
if __name__ == '__main__':
    filtered_ops = list(filter(lambda op: op.dtypes == set(_DYNAMIC_DTYPES), op_db))

    for op in filtered_ops:
        print(OpInfoHelper(op).summary())
