from functools import wraps, partial
from itertools import product, chain
import itertools
import collections
import copy
import operator
import random
import numbers
import unittest

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
    OpInfo, DecorateInfo, SampleInput, sample_inputs_hardshrink_hardtanh,
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

# https://github.com/pytorch/pytorch/pull/61068

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



def sample_inputs_atleast_nd(self, device, dtype, requires_grad):
    inps = []
    for i in range(5):
        inps.append(make_tensor(list(range(i)), device=device, dtype=dtype,
                      requires_grad=requires_grad, low=-1, high=1))

    sample_inputs = []
    for inp in inps:
        sample_inputs.append(SampleInput(inp))

    sample_inputs.append(SampleInput(inps))
    return sample_inputs

for i in range(1, 4):
    additional_op_db.append(
        OpInfo(f'atleast_{i}d',
                aten_name="atleast_{i}d",
                supports_autograd=True,
                sample_inputs_func=sample_inputs_atleast_nd,
                dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                supports_out=False))


