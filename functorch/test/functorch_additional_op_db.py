from functools import wraps, partial
from itertools import product, chain
import itertools
import collections
import copy
import operator
import random
import numbers

import torch
import numpy as np
from torch._six import inf
import collections.abc

from typing import Any, List, Sequence, Tuple, Union

from torch.testing import \
    (make_non_contiguous, floating_types, floating_types_and, complex_types,
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and, all_types_and_complex,
     integral_types_and, all_types)
# from .._core import _dispatch_dtypes
from torch.testing._internal.common_device_type import \
    (skipIf, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfNoCusolver,
     skipCPUIfNoLapack, skipCPUIfNoFFT, skipCUDAIfRocm, precisionOverride, toleranceOverride, tol)
from torch.testing._internal.common_cuda import CUDA11OrLater, SM53OrLater, SM60OrLater
from torch.testing._internal.common_utils import \
    (is_iterable_of_tensors,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     make_fullrank_matrices_with_distinct_singular_values,
     random_symmetric_pd_matrix, make_symmetric_matrices,
     make_symmetric_pd_matrices,
     random_fullrank_matrix_distinct_singular_value,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor, TEST_SCIPY,
     torch_to_numpy_dtype_dict, TEST_WITH_ASAN,
     GRADCHECK_NONDET_TOL,)
import torch.testing._internal.opinfo_helper as opinfo_helper
from torch.testing._internal.common_methods_invocations import OpInfo, SkipInfo, SampleInput

# List of OpInfos that aren't in PyTorch Core yet.
# They are here because we wanted a fast way of writing OpInfos and may not be
# 100% correct (w.r.t. to dtypes and other options).
# TODO: Figure out how to upstream these, delete them when they're upstreamed

additional_op_db = []

# https://github.com/pytorch/pytorch/pull/61971
def sample_inputs_linear(has_bias, self, device, dtype, requires_grad):
    features_options = [[3, 4], [128, 128]]
    batch_options = [
        [], # no batch
        [64],
        [5, 7],
    ]

    sample_inputs = []
    for (in_feat, out_feat), batch_shape in itertools.product(features_options, batch_options):
        input_tensor = make_tensor(batch_shape + [in_feat], device=device,
                                   dtype=dtype, requires_grad=requires_grad,
                                   low=-2, high=2)
        weight = make_tensor([out_feat, in_feat], device=device,
                             dtype=dtype, requires_grad=requires_grad,
                             low=-2, high=2)
        if not has_bias:
            sample_inputs.append(SampleInput(input_tensor, args=(weight,)))
            continue

        bias = make_tensor([out_feat], device=device,
                           dtype=dtype, requires_grad=requires_grad,
                           low=-2, high=2)
        sample_inputs.append(SampleInput(input_tensor, args=(weight, bias)))
    return sample_inputs

additional_op_db.extend([
    OpInfo('nn.functional.linear',
           aten_name='linear',
           variant_test_name='with_bias',
           supports_autograd=True,
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_linear, False),
           supports_out=False),
    OpInfo('nn.functional.linear',
           aten_name='linear',
           variant_test_name='no_bias',
           supports_autograd=True,
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_linear, True),
           supports_out=False),
])

# https://github.com/pytorch/pytorch/pull/61956
def sample_inputs_interpolate(mode, self, device, dtype, requires_grad):
    N, C = 2, 3
    D = 4
    S = 3
    L = 5

    align_corners_options: Tuple[Any, ...] = (None,)
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        align_corners_options = (True, False, None)
    ranks_for_mode = {
        'nearest': [1, 2, 3],
        'linear': [1],
        'bilinear': [2],
        'bicubic': [2],
        'trilinear': [3],
        'area': [1, 2, 3]
    }

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    sample_inputs = []
    for align_corners in align_corners_options:
        for rank in ranks_for_mode[mode]:
            sample_inputs.extend([
                SampleInput(make_tensor(shape(D, rank), device=device, dtype=dtype,
                                        requires_grad=requires_grad, low=-1, high=1),
                            args=(shape(S, rank, False), None, mode, align_corners)),
                SampleInput(make_tensor(shape(D, rank), device=device, dtype=dtype,
                                        requires_grad=requires_grad, low=-1, high=1),
                            args=(shape(L, rank, False), None, mode, align_corners)),
                SampleInput(make_tensor(shape(D, rank), device=device, dtype=dtype,
                                        requires_grad=requires_grad, low=-1, high=1),
                            args=(None, 1.7, mode, align_corners)),
                SampleInput(make_tensor(shape(D, rank), device=device, dtype=dtype,
                                        requires_grad=requires_grad, low=-1, high=1),
                            args=(None, 0.6, mode, align_corners)),
            ])

    return sample_inputs

additional_op_db.extend([
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='nearest',
           supports_autograd=True,
           dtypesIfCPU=floating_types_and(torch.uint8),
           dtypesIfCUDA=floating_types_and(torch.half, torch.uint8),
           sample_inputs_func=partial(sample_inputs_interpolate, 'nearest'),
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='linear',
           supports_autograd=True,
           dtypesIfCUDA=floating_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_interpolate, 'linear'),
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='bilinear',
           supports_autograd=True,
           dtypesIfCUDA=floating_types_and(torch.half),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_interpolate, 'bilinear'),
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='bicubic',
           supports_autograd=True,
           dtypesIfCUDA=floating_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_interpolate, 'bicubic'),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='trilinear',
           supports_autograd=True,
           dtypesIfCUDA=floating_types_and(torch.half),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_interpolate, 'trilinear'),
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='area',
           supports_autograd=True,
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_interpolate, 'area'),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # JIT alias info internal asserts here
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
])
