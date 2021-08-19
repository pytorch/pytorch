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
from torch.testing._internal.common_methods_invocations import (
    OpInfo, SkipInfo, SampleInput, sample_inputs_hardshrink_hardtanh,
    sample_inputs_softmax_variant, S
)

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

# https://github.com/pytorch/pytorch/pull/61068
def sample_inputs_dropout(self, device, dtype, requires_grad):
    samples = []
    dropout_args = [
        (0.6, False, False),
        (1.0, True, False),
        (0.0, True, False)
    ]
    shapes = [(), (2,), (2, 3, 4), (2, 3, 4, 5, 6)]
    for rank in [1, 3, 5]:
        for shape in shapes:
            for args in dropout_args:
                samples.append(SampleInput(make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad, low=-5, high=5), args=args))
    return samples

additional_op_db.extend([
    OpInfo('nn.functional.dropout',
           aten_name="dropout",
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_dropout,
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_gradgrad=False,
           supports_forward_ad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::dropout"]),
])

def sample_inputs_conv2d(has_bias, self, device, dtype, requires_grad, extra_args=(), groups=1):
    in_ch, out_ch = 6, 4
    inp = make_tensor((2, in_ch * groups, 7, 5), device=device, dtype=dtype,
                      requires_grad=requires_grad, low=-1, high=1)
    weight = make_tensor((out_ch * groups, in_ch, 3, 2), device=device, dtype=dtype,
                         requires_grad=requires_grad, low=-1, high=1)
    bias = None
    if has_bias:
        bias = make_tensor((out_ch * groups,), device=device, dtype=dtype,
                           requires_grad=requires_grad, low=-1, high=1)
    return [SampleInput(inp, args=((weight, bias) + extra_args))]

additional_op_db.extend([
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False, extra_args=((2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_padding_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_padding_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False, extra_args=((2, 2), (1, 1))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='strided_padding_dilation_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='strided_padding_dilation_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_groups_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 2), groups=2),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_depthwise_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 6), groups=6),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
])

def sample_inputs_cross_entropy(self, device, dtype, requires_grad, reduction):
    N = 2
    C = 10
    inp = make_tensor((2, C), device=device, dtype=dtype,
                      requires_grad=requires_grad, low=-1, high=1)
    target = torch.randint(0, C, (N,), device=device)
    inp4d = make_tensor((2, C, 4, 5), device=device, dtype=dtype,
                        requires_grad=requires_grad, low=-1, high=1)
    target4d = torch.randint(0, C, (N, 4, 5), device=device)
    weight = make_tensor((C,), device=device, dtype=dtype,
                         low=0.5, high=1)
    sample_inputs = [
        SampleInput(inp, args=(target,), kwargs={'reduction': reduction}),
        SampleInput(inp, args=(target,), kwargs={'ignore_index': 1, 'reduction': reduction}),
        SampleInput(inp, args=(target, weight), kwargs={'ignore_index': 1, 'reduction': reduction}),
    ]
    sample_inputs.extend([
        SampleInput(inp4d, args=(target4d,), kwargs={'reduction': reduction}),
        SampleInput(inp4d, args=(target4d,), kwargs={'ignore_index': 1, 'reduction': reduction}),
        SampleInput(inp4d, args=(target4d, weight), kwargs={'ignore_index': 1, 'reduction': reduction}),
    ])
    return sample_inputs

for reduction in ['mean', 'sum', 'none']:
    additional_op_db.append(
        OpInfo('nn.functional.cross_entropy',
               aten_name="cross_entropy",
               variant_test_name=reduction,
               supports_autograd=True,
               sample_inputs_func=partial(sample_inputs_cross_entropy, reduction=reduction),
               dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
               supports_out=True))


