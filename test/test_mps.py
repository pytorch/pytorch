# Owner(s): ["module: mps"]

import io
import platform
import sys
import math
import random
import unittest
import warnings
import subprocess
import tempfile
import os
import copy
import gc
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict
from torch import inf
from torch.nn import Parameter
from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, run_tests, TestCase, download_file, IS_CI, NoTest,
     skipIfSlowGradcheckEnv, suppress_warnings)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from functools import partial

from torch.testing._internal.common_methods_invocations import (
    op_db,
    DecorateInfo,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    BinaryUfuncInfo,
)
from torch.testing._internal.common_device_type import ops, dtypes, instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_nn import NNTestCase
import numpy as np
import torch
import torch.utils._pytree as pytree
from itertools import product

test_consistency_op_db = copy.deepcopy(op_db)
test_error_inputs_op_db = copy.deepcopy(op_db)

# Copied from `test_ops.py` for the purposes of duplicating `test_numpy_ref`
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

def mps_ops_grad_modifier(ops):
    XFAILLIST_GRAD = {

        # precision issues
        'digamma': [torch.float32],
        'special.polygammaspecial_polygamma_n_0': [torch.float16],
        'polygammapolygamma_n_0': [torch.float16],

        # Unimplemented ops
        '__getitem__': [torch.float16],
        'sgn': [torch.float16, torch.float32],
        '_segment_reduce': [torch.float16, torch.float32],
        'unfold_copy': [torch.float16, torch.float32],  # unfold_backward is not implemented
        'unfold': [torch.float16, torch.float32],
        'sparse.mmreduce': [torch.float32],  # csr not supported
        'unique_consecutive': [torch.float16, torch.float32],
        'special_modified_bessel_i0': [torch.float16, torch.float32],
        'scalar_tensor': [torch.float16, torch.float32],
        'cdist': [torch.float32],
        'masked.scatter': [torch.float16, torch.float32],
        'index_fill': [torch.float16, torch.float32],  # missing `aten::_unique`.
        'aminmax': [torch.float32],
        'polar': [torch.float32],

        # Correctness issues
        'atanh': [torch.float32],

        # Random output
        'exponential': [torch.float16, torch.float32],

        # CPU errors
        # derivative for aten::floor_divide is not implemented on CPU
        'floor_divide': [torch.float16, torch.float32],
        # derivative for aten::narrow_copy is not implemented on CPU
        'narrow_copy': [torch.float16, torch.float32],
        # derivative for aten::_histogramdd_from_bin_cts is not implemented on CPU
        'histogramdd': [torch.float16, torch.float32],
        # derivative for aten::histogram is not implemented
        'histogram': [torch.float16, torch.float32],
        # 'bool' object is not iterable
        'allclose': [torch.float16, torch.float32],
        'equal': [torch.float16, torch.float32],
        # 'float' object is not iterable
        'item': [torch.float16, torch.float32],
        # "mse_backward_cpu_out" not implemented for 'Half'
        'nn.functional.mse_loss': [torch.float16],
        # "smooth_l1_backward_cpu_out" not implemented for 'Half'
        'nn.functional.smooth_l1_loss': [torch.float16],
        # cpu error: grad requires non-empty inputs
        'randn': [torch.float16, torch.float32],
        'signal.windows.bartlett': [torch.float32],
        'signal.windows.blackman': [torch.float32],
        'signal.windows.cosine': [torch.float32],
        'signal.windows.exponential': [torch.float32],
        'signal.windows.gaussian': [torch.float32],
        'signal.windows.general_cosine': [torch.float32],
        'signal.windows.general_hamming': [torch.float32],
        'signal.windows.hamming': [torch.float32],
        'signal.windows.hann': [torch.float32],
        'signal.windows.kaiser': [torch.float32],
        'signal.windows.nuttall': [torch.float32],
        'eye': [torch.float16, torch.float32],

        # trunc_tensor not working properly for float16
        'divtrunc_rounding': [torch.float16],
        'fmod': [torch.float16],

        # round not working properly for float16
        'round': [torch.float16],
    }

    MACOS_12_3_XFAILLIST_GRAD = {
        # Unsupported Border padding mode, forward pass success as fallback to cpu
        'grid_sampler_2d': [torch.float32],
        # Unimplemented
        'logaddexp2': [torch.float32],

        # The result of pow(9 , 8) is showing 43046716, whereas it should've been 43046721.
        # fixed in macOS 13. We are not raising error.
        '__rpow__': [torch.float32],
        'pow': [torch.float32],
    }

    MACOS_BEFORE_13_3_XFAILLIST_GRAD = {
        # Failures due to precision issues (due to fast-math). These has been fixed in MacOS 13.3+
        'masked.softmin': [torch.float32],
        'masked.softmax': [torch.float32],
        'masked.log_softmax': [torch.float32],

        # Unsupported Border padding mode, forward pass success as fallback to cpu
        'grid_sampler_2d': [torch.float32],

        # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
        # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
        # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
        # Running `msort` with stable `sort` passes.
        'msort': [torch.float16],

        # The result of pow(9 , 8) is showing 43046716, whereas it should've been 43046721.
        # fixed in macOS 13. We are not raising error.
        'pow': [torch.float32],
        '__rpow__': [torch.float32],

        # See https://github.com/pytorch/pytorch/issues/106112 for more information
        'cumprod': [torch.float32],
    }

    XPASSLIST_GRAD = {
        'nn.functional.pairwise_distance': [torch.float16],
        # failed assertion `destination datatype must be fp32'
        'nn.functional.conv1d': [torch.float16],
        'nn.functional.conv2d': [torch.float16],
        'nn.functional.conv3d': [torch.float16],
        'nn.functional.conv_transpose1d': [torch.float16],
        'nn.functional.conv_transpose2d': [torch.float16],
        'nn.functional.conv_transpose3d': [torch.float16],
    }

    MACOS_13_3_XFAILLIST_GRAD = {
        # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
        # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
        # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
        # Running `msort` with stable `sort` passes.
        'msort': [torch.float16],
    }

    def addDecorator(op, d) -> None:
        op.decorators = list(op.decorators) if op.decorators is not None else []
        op.decorators.append(d)

    for op in ops:
        key = op.name + op.variant_test_name
        if key in XFAILLIST_GRAD:
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=XFAILLIST_GRAD[key]))

        if key in XPASSLIST_GRAD:
            addDecorator(op, DecorateInfo(
                         unittest.skip,
                         dtypes=XPASSLIST_GRAD[key]))

        if key in MACOS_12_3_XFAILLIST_GRAD and (not torch.backends.mps.is_macos13_or_newer()):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_12_3_XFAILLIST_GRAD[key]))

        if key in MACOS_BEFORE_13_3_XFAILLIST_GRAD and (torch.backends.mps.is_macos13_or_newer() and product_version < 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_BEFORE_13_3_XFAILLIST_GRAD[key]))

        if key in MACOS_13_3_XFAILLIST_GRAD and (product_version >= 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_13_3_XFAILLIST_GRAD[key]))
        yield op

def mps_ops_modifier(ops):
    # Supported complex OPS
    SUPPORTED_COMPLEX_OPS = {
        '__radd__',
        '__rmul__',
        'add',
        'atleast_1d',
        'atleast_2d',
        'atleast_3d',
        'clone',
        'contiguous',
        'empty',
        'empty_permuted',
        'empty_strided',
        'eye',
        'flatten',
        'full',
        'imag',
        'isfinite',
        'isinf',
        'isreal',
        'item',
        'kron',
        'linspace',
        'logspace',
        'linspacetensor_overload',
        'logspacetensor_overload',
        'mul',
        'nn.functional.feature_alpha_dropoutwithout_train',
        'nn.functional.unfold',
        'ones',
        'outer',
        'positive',
        'randn',
        'ravel',
        'real',
        'reshape_as',
        'reshape',
        'resolve_conj',
        'resolve_neg',
        'scalar_tensor',
        'sgn',
        'split',
        'squeeze',
        'squeezemultiple',
        'sub',
        't',
        'unflatten',
        'unsafe_split',
        'unsqueeze',
        'view_as',
        'view_as_real',
        'view',
        'vsplit',
        'zero_',
        'zeros',
    }
    # Those ops worked on MacOS12, but broken on MacOS13, see https://github.com/pytorch/pytorch/issues/85758
    MACOS_12_3_XFAILLIST = {
        # Top 60
        # expected failures
        # The result of pow(9 , 8) is showing 43046716, whereas it should've been 43046721.
        # fixed in macOS 13.3. Currently error is not raised.
        'pow': [torch.int16, torch.int64, torch.uint8, torch.int8],
        # expected failures
        '__rpow__': [torch.uint8, torch.int8],

        # Failures due to precision issues (due to fast-math). These has been fixed in MacOS 13.3+
        'cdist': [torch.float32],
        'tan': [torch.uint8, torch.float32],

        # Data type support starts from macOS 13
        'nn.functional.avg_pool1d': [torch.int64],
        'nn.functional.avg_pool2d': [torch.int64],
        'nn.functional.local_response_norm': [torch.int64],
        '__radd__': [torch.uint8],
        '__rdiv__': [torch.uint8],
        '__rmul__': [torch.uint8],
        'abs': [torch.uint8],
        'acos': [torch.uint8],
        'acosh': [torch.uint8],
        'add': [torch.uint8],
        'asin': [torch.uint8],
        'asinh': [torch.uint8],
        'atan': [torch.uint8],
        'atanh': [torch.uint8],
        'ceil': [torch.uint8],
        'corrcoef': [torch.uint8],
        'cos': [torch.uint8],
        'cosh': [torch.uint8],
        'cov': [torch.uint8],
        'cumulative_trapezoid': [torch.uint8],
        'deg2rad': [torch.uint8],
        'diff': [torch.uint8],
        'eq': [torch.uint8],
        'equal': [torch.uint8],
        'erf': [torch.uint8],
        'exp2': [torch.uint8],
        'exp': [torch.uint8],
        'expm1': [torch.uint8],
        'floor': [torch.uint8],
        'fmax': [torch.uint8],
        'fmin': [torch.uint8],
        'fmod': [torch.uint8],
        'ge': [torch.uint8],
        'gt': [torch.uint8],
        'isclose': [torch.uint8],
        'isnan': [torch.uint8],
        'kron': [torch.uint8],
        'le': [torch.uint8],
        'log10': [torch.uint8],
        'log1p': [torch.uint8],
        'log2': [torch.uint8],
        'log': [torch.uint8],
        'logical_and': [torch.uint8],
        'logical_or': [torch.uint8],
        'logical_xor': [torch.uint8],
        'logit': [torch.uint8],
        'lt': [torch.uint8],
        'masked.mean': [torch.uint8],
        'masked.std': [torch.uint8],
        'masked.var': [torch.uint8],
        'maximum': [torch.uint8],
        'minimum': [torch.uint8],
        'mul': [torch.uint8],
        'ne': [torch.uint8],
        'neg': [torch.uint8],
        'nn.functional.cosine_embedding_loss': [torch.uint8],
        'nn.functional.margin_ranking_loss': [torch.uint8],
        'nn.functional.poisson_nll_loss': [torch.uint8],
        'nn.functional.softsign': [torch.uint8],
        'nn.functional.tanhshrink': [torch.uint8],
        'nn.functional.triplet_margin_loss': [torch.uint8],
        'nn.functional.triplet_margin_with_distance_loss': [torch.uint8],
        'nn.functional.pairwise_distance': [torch.uint8],
        'outer': [torch.uint8],
        'rad2deg': [torch.uint8],
        'reciprocal': [torch.uint8],
        'remainder': [torch.uint8],
        'round': [torch.uint8],
        'rsqrt': [torch.uint8],
        'sigmoid': [torch.uint8],
        'sign': [torch.uint8],
        'signbit': [torch.uint8],
        'sin': [torch.uint8],
        'sinh': [torch.uint8],
        'special.ndtr': [torch.uint8],
        'sqrt': [torch.uint8],
        'sub': [torch.uint8],
        'tanh': [torch.uint8],
        'trapezoid': [torch.uint8],
        'trapz': [torch.uint8],
        'true_divide': [torch.uint8],
        'trunc': [torch.uint8],
        'xlogy': [torch.uint8],
        'minbinary': [torch.uint8],
        'maxbinary': [torch.uint8],
        'divtrunc_rounding': [torch.uint8],
        'divfloor_rounding': [torch.uint8],
        'divno_rounding_mode': [torch.uint8],
        'floor_divide': [torch.uint8],
        'ldexp': [torch.uint8],
        # square internally calls into power, and will type cast to int64, which supports starting from macOS 13
        'square': [torch.bool, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # cpu not giving nan for x/0.0
        'atan2': [torch.bool, torch.float16, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # inconsistency errors between cpu and mps, max seen atol is 2
        'nn.functional.interpolatebilinear': [torch.uint8],
    }

    MACOS_BEFORE_13_3_XFAILLIST = {
        # Failure due to precision issues (still present on 13.3+) as well as non-standard behavior of
        # cpu ops for the negative integers.
        # Example for torch.polygamma(1, tensor([-0.9, -1.0], dtype=torch.float32)):
        # - CPU output: tensor([102.668, 1.129e+15])
        # - MPS output: tensor([102.6681, inf])
        # In the latter case, inf is probably correct (this is what scipy does).
        'polygamma': [torch.float32, torch.uint8],
        'polygammapolygamma_n_0': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_2': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_1': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_3': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_4': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'special.polygamma': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'special.polygammaspecial_polygamma_n_0': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],

        # Failures due to precision issues (due to fast-math). These has been fixed in MacOS 13.3+
        'tan': [torch.float32],
        'cdist': [torch.float32],

        # CPU Error: cpu not giving nan for x/0.0
        'atan2': [torch.bool, torch.float16, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # test blow pass on macOS 12 as it falls back to cpu
        # Argsort case using duplicate indices (undefined behaviour):
        #  - CPU output: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], devuce='cpu')
        #  - MPS output: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
        # Elements from index 30 and 5133 are both equal.
        # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
        'argsort': [torch.float16, torch.int8, torch.uint8, torch.bool],
        # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
        # The values of the sorted tensor match the CPU, but in case of the returned indices this results in undefined behaviour.
        'sort': [torch.int8, torch.uint8, torch.bool, torch.float16],
        # Unsupported dtypes
        'cumsum': [torch.int64],
        'cumprod': [torch.int64],
        'cumulative_trapezoid': [torch.int64],
        'masked.cumsum': [torch.int64],
        'masked.cumprod': [torch.int64],
        'linalg.vander': [torch.int64],
    }

    MACOS_AFTER_13_1_XFAILLIST = {
        # before macOS 13.2 it falls back to cpu and pass the forward pass
        'grid_sampler_2d': [torch.float32],  # Unsupported Border padding mode
        # inconsistency errors between cpu and mps, max seen atol is 2
        'nn.functional.interpolatebilinear': [torch.uint8],
    }

    MACOS_13_3_XFAILLIST = {
        # Failure due to precision issue for fp16
        # on both cpu and mps there are test cases that might produce inf result
        # 'nn.functional.pairwise_distance': [torch.float16],

        # test blow pass on macOS 12 as it falls back to cpu
        # Argsort case using duplicate indices (undefined behaviour):
        #  - CPU output: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], devuce='cpu')
        #  - MPS output: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
        # Elements from index 30 and 5133 are both equal.
        # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
        'argsort': [torch.float16, torch.int8, torch.uint8, torch.bool],
        # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
        # The values of the sorted tensor match the CPU, but in case of the returned indices this results in undefined behaviour.
        'sort': [torch.int8, torch.uint8, torch.bool, torch.float16],

        # Failure due to precision issues as well as non-standard behavior of cpu ops for the
        # negative integers. Example for torch.polygamma(1, tensor([-0.9, -1.0], dtype=torch.float32)):
        # - CPU output: tensor([102.668, 1.129e+15])
        # - MPS output: tensor([102.6681, inf])
        # In the latter case, inf is probably correct (this is what scipy does).
        'polygamma': [torch.float32, torch.uint8],
        'polygammapolygamma_n_0': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_2': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_1': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_3': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'polygammapolygamma_n_4': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'special.polygamma': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
        'special.polygammaspecial_polygamma_n_0': [torch.float32, torch.int16, torch.int32, torch.int64, torch.int8],
    }

    # Those ops are not expected to work
    UNIMPLEMENTED_XFAILLIST = {
        # Failures due to lack of op implementation on MPS backend
        'login': None,
        'log_sigmoid': None,
        'log_sigmoid_forward': None,
        'linalg.eig': None,
        'linalg.eigvals': None,
        'fft.fft': None,
        'fft.fft2': None,
        'fft.fftn': None,
        'fft.hfft': None,
        'fft.hfft2': None,
        'fft.hfftn': None,
        'fft.ifft': None,
        'fft.ifft2': None,
        'fft.ifftn': None,
        'fft.ihfft': None,
        'fft.ihfft2': None,
        'fft.ihfftn': None,
        'fft.irfft': None,
        'fft.irfft2': None,
        'fft.irfftn': None,
        'fft.rfft': None,
        'fft.rfft2': None,
        'fft.rfftn': None,
        'put': None,
        'stft': None,
        'nn.functional.conv_transpose3d': None,
        'rounddecimals_neg_3': None,
        'rounddecimals_3': None,
        'rounddecimals_0': None,
        '__rsub__': None,
        'angle': None,
        'bucketize': None,
        'cauchy_': None,
        'cauchy': None,
        'cholesky': None,
        'cholesky_inverse': None,
        'cholesky_solve': None,
        'cummax': None,
        'cummin': None,
        'erfc': None,
        'frexp': None,
        'gcd': None,
        'geqrf': None,
        'nn.functional.grid_sample': None,  # Unsupported Border padding mode
        'heaviside': None,
        'i0': None,
        'igamma': None,
        'igammac': None,
        'index_copy': None,
        'index_reduce': None,
        'isin': None,
        'isneginf': None,
        'isposinf': None,
        'kthvalue': None,
        'lcm': None,
        'linalg.cholesky': None,
        'linalg.cholesky_ex': None,
        'linalg.cond': None,
        'linalg.detsingular': None,
        'linalg.det': None,
        'linalg.eigh': None,
        'linalg.eigvalsh': None,
        'linalg.householder_product': None,
        'linalg.ldl_factor': None,
        'linalg.ldl_factor_ex': None,
        'linalg.ldl_solve': None,
        'linalg.lstsq': None,
        'linalg.lstsqgrad_oriented': None,
        'linalg.lu': None,
        'linalg.lu_factor': None,
        'linalg.lu_factor_ex': None,
        'linalg.lu_solve': None,
        'linalg.matrix_norm': [torch.float32],
        'linalg.norm': [torch.float32],
        'linalg.normsubgradients_at_zero': [torch.float32],
        'linalg.qr': None,
        'linalg.slogdet': None,
        'linalg.solve': None,
        'linalg.solve_ex': None,
        'linalg.svdvals': None,
        'linalg.tensorsolve': None,
        'linalg.vecdot': None,
        'logcumsumexp': None,
        'logdet': None,
        'lu': None,
        'lu_solve': None,
        'lu_unpack': None,
        'masked.median': None,
        'matrix_exp': None,
        'mode': None,
        'nanquantile': None,
        'nanmedian': None,
        'native_dropout_backward': None,
        'nextafter': None,
        'normnuc': None,
        'nn.functional.fractional_max_pool2d': None,
        'nn.functional.fractional_max_pool3d': None,
        'nn.functional.adaptive_avg_pool3d': None,
        'nn.functional.adaptive_max_pool3d': None,
        'nn.functional.interpolatearea': None,
        'nn.functional.interpolatebicubic': None,
        'nn.functional.interpolatelinear': None,
        'nn.functional.interpolatetrilinear': None,
        # TODO: max_pool2d for integral types fails the numerical test
        'nn.functional.max_pool2d': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'nn.functional.max_unpool1dgrad': None,
        'nn.functional.max_unpool2dgrad': None,
        'nn.functional.max_unpool3dgrad': None,
        'nn.functional.avg_pool3d': None,
        'nn.functional.ctc_loss': None,
        'nn.functional.embedding_bag': None,
        'nn.functional.hardshrink': None,
        'nn.functional.max_pool3d': None,
        'nn.functional.max_unpool1d': None,
        'nn.functional.max_unpool2d': None,
        'nn.functional.max_unpool3d': None,
        'nn.functional.multi_margin_loss': None,
        'nn.functional.multilabel_margin_loss': None,
        'nn.functional.pdist': None,
        'nn.functional.rrelu': None,
        'nn.functional.softshrink': None,
        'nn.functional.norm': None,
        'ormqr': None,
        'pca_lowrank': None,
        'pinverse': None,
        'qr': None,
        'quantile': None,
        'rsub': None,
        'scatter_reduceamax': None,
        'scatter_reduceamin': None,
        'scatter_reducemin': None,
        'scatter_reducemean': None,
        'scatter_reduceprod': None,
        'scatter_reducesum': None,
        'searchsorted': None,
        'segment_reduce': None,
        '_segment.reduce': None,
        'segment.reduce': None,
        'segment_reduce_offsets': None,
        '_segment_reduce_offsets': None,
        '_segment_reduce_lengths': None,
        '_segment_reducelengths': None,
        '_segment_reduceoffsets': None,
        'sinc': None,
        'sparse.mm': None,
        'sparse.mmreduce': None,
        'special.airy_ai': None,
        'special.bessel_j0': None,
        'special.bessel_j1': None,
        'special.bessel_y0': None,
        'special.bessel_y1': None,
        'special.chebyshev_polynomial_t': None,
        'special.chebyshev_polynomial_u': None,
        'special.entr': None,
        'special.erfcx': None,
        'special.hermite_polynomial_h': None,
        'special.hermite_polynomial_he': None,
        'special.i0e': None,
        'special.i1': None,
        'special.i1e': None,
        'special.laguerre_polynomial_l': None,
        'special.log_ndtr': None,
        'special.modified_bessel_i0': None,
        'special.modified_bessel_i1': None,
        'special.modified_bessel_k0': None,
        'special.modified_bessel_k1': None,
        'special.ndtri': None,
        'special.scaled_modified_bessel_k0': None,
        'special.scaled_modified_bessel_k1': None,
        'special.spherical_bessel_j0': None,
        'special.xlog1py': None,
        'special.zeta': None,
        'std_mean': None,
        'std_meanunbiased': None,
        'svd_lowrank': None,
        'symeig': None,
        'take': None,
        'to': None,
        'to_sparse': None,
        'unique': None,
        'vdot': None,
        'segment_reduce_': None,
        '_upsample_bilinear2d_aa': None,
        'geometric' : None,
        'geometric_': None,
        'log_normal_': None,
        'log_normal': None,
        'bfloat16': None,
        'cdouble': None,
        'cfloat': None,
        'complex': None,
        'double': None,
        'chalf': None,
        'nn.functional.softminwith_dtype': None,
        'log_softmaxwith_dtype': None,
        'softmaxwith_dtype': None,
        'float_power': None,
        'full_like': None,
        'linalg.matrix_rank': None,
        'linalg.matrix_rankhermitian': None,
        'linalg.pinv': None,
        'linalg.pinvhermitian': None,
        'nonzero_static': None,

        # MPS: input sizes must be divisible by output sizes
        'nn.functional.adaptive_avg_pool1d': None,
        'nn.functional.adaptive_avg_pool2d': None,

        # Unsupported dtypes
        # bmm is not supported for integral types
        'nn.functional.bilinear': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        # Cannot convert a MPS Tensor to float64 dtype. The tensors
        # input data is created with double in common_methods_invocations.py
        'nn.functional.batch_norm': [torch.float32],
        'ones_like': None,
        'zeros_like': None,

        # Convolution for integral types is not supported on MPS
        'nn.functional.conv1d': [torch.int64],
        'nn.functional.conv2d': [torch.int64],
        'nn.functional.conv_transpose1d': [torch.int64],
        'nn.functional.conv_transpose2d': [torch.int64],

        # Unsupported dtypes
        'dot': [torch.int64],
        'index_add': [torch.int64],
        'log1p': [torch.int64],
        'sigmoid': [torch.int64],
        'atan2': [torch.int64],

        # GEMM on MPS is not supported for integral types
        'nn.functional.linear': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        '__rmatmul__': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'addmmdecomposed': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'addbmm': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'addmm': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'addmv': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'baddbmm': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'mm': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'bmm': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'einsum': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'inner': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'linalg.multi_dot': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'matmul': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'mat': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'mv': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'tensordot': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # new_zeros/new_ones: Cannot convert a MPS Tensor to float64 dtype as
        # the MPS framework doesn't support float64
        'new_zeros': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'new_ones': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'new_full': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        # returned output on CPU is float64
        'bincount': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # trunc_tensor not working properly for float16
        'divtrunc_rounding': [torch.float16],
        'fmod': [torch.float16],

        # round not working properly for float16
        'round': [torch.float16],
    }

    UNDEFINED_XFAILLIST = {
        # Top 60 operators
        # topk fails with duplicate indices
        'topk': [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # Failures due to random output that they generate using
        # Philox engine causing mismatch with CPU results
        'multinomial': [torch.float32],  # random results
        'uniform': [torch.float16, torch.float32],
        'rand_like': [torch.float16, torch.float32],
        'randint_like': [torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'randn_like': [torch.float16, torch.float32],
        'bernoulli': [torch.float32],
        'exponential': [torch.float16, torch.float32],
        'nn.functional.feature_alpha_dropoutwith_train': [torch.float32],
        'normal': [torch.float16, torch.float32, torch.float16, torch.float32],
        'normalin_place': [torch.float16, torch.float32],
        'normalnumber_mean': [torch.float16, torch.float32],
        'nn.functional.alpha_dropout': [torch.float32],
        'nn.functional.dropout': [torch.float32],
        'nn.functional.dropout2d': [torch.float32],
        'nn.functional.dropout3d': [torch.float32],
        'nn.functional.multi_head_attention_forward': [torch.float32],

        # duplicate indices are used in the testcase - undefined behaviour
        'index_put': None,
        # zero to negative integer powers are undefined
        '__rpow__': [torch.int8, torch.int16, torch.int32, torch.int64],
        'resize_': [torch.float16, torch.float32],
        'resize_as_': [torch.float16, torch.float32],

        # CPU Errors:
        'addr': [torch.bool, torch.int16, torch.int32,
                 torch.int64, torch.uint8, torch.int8],  # "addmv_impl_cpu" not implemented for 'Half'
        'as_stridedpartial_views': [torch.bool, torch.float16, torch.float32, torch.int16,
                                    torch.int32, torch.int64, torch.uint8, torch.int8],  # cpu result off, showing random values
        'as_strided_partial_views': [torch.bool, torch.float16, torch.float32, torch.int16,
                                     torch.int32, torch.int64, torch.uint8, torch.int8],  # cpu result off, showing random values

        # random results
        # mps vs cpu:
        # Mismatched elements: 40 / 96 (41.7%)
        # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
        # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
        # cuda(2.0.0.dev20230301+cu117) vs cpu:
        # Mismatched elements: 56 / 96 (58.3%)
        # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
        # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
        'nn.functional.scaled_dot_product_attention': [torch.float32],

        # Failures due to casting negative float to uint8 is undefined
        'byte': [torch.float16, torch.float32],
    }

    EMPTY_OPS_SKIPLIST = {
        # Fill tensors with uninitialized data, causing mismatch with CPU.
        # They occasionally match, thus skipping them.
        # See https://github.com/pytorch/pytorch/issues/100175
        'new_empty': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'new_empty_strided': [torch.bool, torch.float16, torch.float32, torch.int16,
                              torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_strided': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        # CPU: empty is returning all 0's and there is a mismatch with MPS
        # allocation (MacOS 13). According to
        # https://pytorch.org/docs/2.0/generated/torch.empty.html
        'empty': [torch.bool, torch.float16, torch.float32, torch.int16,
                  torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_like': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_permuted': [torch.bool, torch.float16, torch.float32, torch.int16,
                           torch.int32, torch.int64, torch.uint8, torch.int8],
    }

    def addDecorator(op, d) -> None:
        op.decorators = list(op.decorators) if op.decorators is not None else []
        op.decorators.append(d)

    for op in ops:
        key = op.name + op.variant_test_name
        if key in EMPTY_OPS_SKIPLIST:
            addDecorator(op, DecorateInfo(
                         unittest.skip("Skipping empty ops."),
                         dtypes=EMPTY_OPS_SKIPLIST[key]))
        for xfaillist in [UNIMPLEMENTED_XFAILLIST, UNDEFINED_XFAILLIST]:
            if key in xfaillist:
                addDecorator(op, DecorateInfo(
                             unittest.expectedFailure,
                             dtypes=xfaillist[key]))

        if key in MACOS_BEFORE_13_3_XFAILLIST and (torch.backends.mps.is_macos13_or_newer() and product_version < 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_BEFORE_13_3_XFAILLIST[key]))

        if key in MACOS_AFTER_13_1_XFAILLIST and torch.backends.mps.is_macos13_or_newer(2):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_AFTER_13_1_XFAILLIST[key]))

        if key in MACOS_13_3_XFAILLIST and (product_version >= 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_13_3_XFAILLIST[key]))

        if key in MACOS_12_3_XFAILLIST and (not torch.backends.mps.is_macos13_or_newer()):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_12_3_XFAILLIST[key]))
        # If ops is not supported for complex types, expect it to fail
        if key not in SUPPORTED_COMPLEX_OPS:
            addDecorator(op, DecorateInfo(unittest.expectedFailure, dtypes=[torch.complex32, torch.complex64]))
        yield op

def mps_ops_error_inputs_modifier(ops):
    # Error input samples do not take a dtype argument.
    XFAILLIST = {
        # Exceptions are not raised
        '__rmod__',
        '__rsub__',
        '__rpow__',
        'bernoulli',
        'clamp_max',
        'clamp_min',
        'masked_scatter',

        # unsupported float64 dtype
        'cat',
        'complex',
        'multinomial',
        'nn.functional.conv1d',
        'nn.functional.conv2d',
        'gather',
        'scatter',
        'scatter_add',

        # unsupported complex dtypes
        'masked_fill',
        'fft.hfft',
        'fft.irfft',

        # MPS does not support tensor dimensions > 16
        'amax',
        'amin',
        'aminmax',

        # memory overlapping checks
        'index_select',

        # unimplemented
        'logcumsumexp',
    }

    def addDecorator(op, d) -> None:
        op.decorators = list(op.decorators) if op.decorators is not None else []
        op.decorators.append(d)

    for op in ops:
        if op.error_inputs_func is None:
            continue
        key = op.name + op.variant_test_name
        if key in XFAILLIST:
            addDecorator(op, DecorateInfo(unittest.expectedFailure))
        yield op

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    NNTestCase = NoTest  # noqa: F811

product_version = float('.'.join(platform.mac_ver()[0].split('.')[:2]) or -1)

# Determine whether to enable MPS memory leak check (uses same code as CUDA).
TEST_MPS_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_MPS_MEM_LEAK_CHECK', '0') == '1'

def skipMPSMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_mps_memory_leak_check', True):
            fn._do_mps_memory_leak_check = not condition
        return fn
    return dec

class MpsMemoryLeakCheck:
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

    def __enter__(self):
        # Performs a gc if required (required if any memory is held)
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > 0:
            gc.collect()
            torch.mps.empty_cache()

        # Acquires caching allocator and driver statistics before the test is run
        self.caching_allocator_before = torch.mps.current_allocated_memory()
        self.driver_before = torch.mps.driver_allocated_memory()

    def __exit__(self, exec_type, exec_value, traceback):
        # Don't check for leaks if an exception was thrown
        if exec_type is not None:
            return
        # Compares caching allocator before/after statistics
        # An increase in allocated memory is a discrepancy indicating a possible memory leak
        discrepancy_detected = False
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > self.caching_allocator_before:
            discrepancy_detected = True

        # Short-circuits if no discrepancy detected
        if not discrepancy_detected:
            return
        # Validates the discrepancy persists after garbage collection and
        # is confirmed by the driver API
        gc.collect()
        torch.mps.empty_cache()

        discrepancy_detected = True
        # Query memory multiple items to ensure leak was not transient
        for n in range(3):
            caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
            driver_mem_allocated = torch.mps.driver_allocated_memory()

            caching_allocator_discrepancy = False
            driver_discrepancy = False

            if caching_allocator_mem_allocated > self.caching_allocator_before:
                caching_allocator_discrepancy = True

            if driver_mem_allocated > self.driver_before:
                driver_discrepancy = True

            if not(caching_allocator_discrepancy or driver_discrepancy):
                # Leak was false positive, exit loop
                discrepancy_detected = False
                break

        if caching_allocator_discrepancy and not driver_discrepancy:
            # Just raises a warning if the leak is not validated by the driver API
            msg = ("MPS caching allocator reports a memory leak not "
                   "verified by the driver API in {}! "
                   "Caching allocator allocated memory was {} and is now reported as {}. "
                   "MPS driver allocated memory was {} and is now {}.").format(
                self.name, self.caching_allocator_before,
                caching_allocator_mem_allocated, self.driver_before, driver_mem_allocated)
            warnings.warn(msg)
        elif caching_allocator_discrepancy and driver_discrepancy:
            # A caching allocator discrepancy validated by the driver API is a failure
            msg = ("MPS driver API confirmed a leak in {}! "
                   "Caching allocator allocated memory was {} and is now reported as {}. "
                   "MPS driver allocated memory was {} and is now {}.").format(
                self.name, self.caching_allocator_before, caching_allocator_mem_allocated,
                self.driver_before, driver_mem_allocated)

            raise RuntimeError(msg)

# Expand TestCase class with Memory Leak Detection on MPS device
class TestCaseMPS(TestCase):
    _do_mps_memory_leak_check = True

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do MPS memory check.
            if TEST_MPS_MEM_LEAK_CHECK:
                if self._do_mps_memory_leak_check:
                    self.wrap_with_mps_policy(method_name, self.assertLeaksNoMpsTensors)

    def assertLeaksNoMpsTensors(self, name=None):
        name = self.id() if name is None else name
        return MpsMemoryLeakCheck(self, name)

    def wrap_with_mps_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        setattr(self, method_name, super().wrap_method_with_policy(test_method, policy))

    # checks for leaks even if TEST_MPS_MEM_LEAK_CHECK is 0
    def wrap_with_mps_memory_check(self, method):
        return super().wrap_method_with_policy(method, self.assertLeaksNoMpsTensors)

class TestMemoryLeak(TestCaseMPS):
    def test_mps_memory_leak_detection(self):
        l = []

        @self.wrap_with_mps_memory_check
        def no_leak():
            pass

        # Trigger an intentional memory leak
        @self.wrap_with_mps_memory_check
        def leak_gpu0():
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("mps")))

        no_leak()

        # check if a runtime error for memory leak was emitted which would
        # confirm whether memory leak detection worked successfully or not.
        with self.assertRaisesRegex(RuntimeError, r"MPS driver API confirmed .+"):
            leak_gpu0()


class TestPixelShuffle(TestCaseMPS):
    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None, is_contiguous=True):

            def generate_input():
                # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
                channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
                height = random.randint(5, 10)
                width = random.randint(5, 10)

                if num_input_dims == 1:
                    input = torch.rand(channels, requires_grad=True, device='mps')
                    assert is_contiguous
                elif num_input_dims == 2:
                    input = torch.rand(width, height, requires_grad=True, device='mps').T
                    if is_contiguous:
                        input = input.contiguous()
                else:
                    batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                    input = torch.rand(*batch_sizes, channels, width, height, requires_grad=True, device='mps')
                    input = input.transpose(-1, -2)
                    if is_contiguous:
                        input = input.contiguous()

                if not is_contiguous and len(input.reshape(-1)) > 0:
                    assert not input.is_contiguous()

                input = input.detach().clone()
                input.requires_grad = True
                return input

            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            input = generate_input()

            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)

                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True, device='mps')
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True, device='mps')
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True, device='mps')

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            is_contiguous_check = [True, False] if num_input_dims > 1 else [True]
            for is_contiguous in is_contiguous_check:
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, valid_channels_dim=False, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=0, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=-2, is_contiguous=is_contiguous
                )

                # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

class MPSReluTest(TestCaseMPS):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)

    def testNpRelu(self):
        torch.testing.assert_close(
            np.array([[0., 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
            self._npRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]])))

    def _testRelu(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=False)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)
        # Inplace Relu modifies the initial input and it should match the output of Relu
        self.assertEqual(np_relu, py_tensor.to("cpu"))

    def testNumbersCPU(self):
        for t in [np.int32]:
            # Force execution on CPU even if a GPU kernel is available for the type.
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")

    def testNumbersGPU(self):
        for t in [np.float16, np.float32]:
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")

class MatmulTest(TestCaseMPS):
    def _helper(self, shape_tensor_1, shape_tensor_2, expand_tensor_1_shape=None, expand_tensor_2_shape=None):
        if expand_tensor_1_shape:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps").expand(expand_tensor_1_shape)
        else:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps")

        if expand_tensor_2_shape:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps").expand(expand_tensor_2_shape)
        else:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps")

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")

        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_vector_x_vector(self):
        # uses `dot`
        self._helper(3, 3)

    def test_matrix_x_vector(self):
        # uses `addmv`
        self._helper((3, 4), 4)

    def test_batched_matrix_x_broadcasted_vector(self):
        self._helper((10, 3, 4), 4)

    def test_batched_matrix_x_batched_matrix(self):
        # uses `bmm.out`
        self._helper((10, 3, 4), (10, 4, 5))

    def test_batched_matrix_x_broadcasted_matrix(self):
        self._helper((10, 3, 4), (4, 5))


class MPSLeakyReluTest(TestCaseMPS):
    def _npLeakyRelu(self, np_features, negative_slope=0.1):
        return np.maximum(np_features, negative_slope * np_features).astype(np_features.dtype)

    def testNpLeakyRelu(self):
        torch.testing.assert_close(
            np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                      [0.1, -0.03, 0.5, -0.07, 0.9]]),
            self._npLeakyRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]]),
                negative_slope=0.1))

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersCPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.2,
                device="cpu")


class TestAvgPool(TestCaseMPS):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: F.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool2d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertTrue(not torch.isnan(y).any())
        y = torch.nn.functional.avg_pool2d(
            x.to('mps'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertTrue(not torch.isnan(y).any())


class TestMPS(TestCaseMPS):
    def test_exp(self, device="mps", dtype=torch.float):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            b = torch.arange(18, device="cpu") / 3 * math.pi
            a = torch.tensor(v, dtype=dtype, device="cpu") * b
            a = a.to(dtype).to("mps")
            self.compare_with_numpy(torch.exp, np.exp, a)

    def test_exp1(self, device="mps", dtype=torch.float):
        input = torch.tensor([-0.1, 3.0, -0.9]).to('mps')
        output = torch.exp(input).to('cpu')

    def test_exp_strided_output(self):
        x = torch.rand((256, 10), device='mps')
        x_cpu = x.to("cpu")

        x = x.permute(1, 0)
        x_cpu = x_cpu.permute(1, 0)

        res = x.exp()
        res_cpu = x_cpu.exp()
        self.assertEqual(res, res_cpu)

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersGPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.1,
                device="mps")

    def test_fill(self):

        def helper(val, shape):
            tensor = torch.zeros(shape, device='mps')
            tensor_mps = tensor.fill_(val)
            tensor_mps = torch.tanh(tensor_mps)

            tensor_0 = torch.zeros(shape, device='cpu')
            tensor_cpu = tensor_0.fill_(val)
            tensor_cpu = torch.tanh(tensor_cpu)

            self.assertEqual(tensor_mps, tensor_cpu)

        helper(0, [1024])
        helper(0.2, [2, 3])

    def test_fill_storage_offset(self):
        shape = [2, 10]
        val = 0.2
        tensor = torch.ones(shape, device="mps")
        tensor_mps = tensor[:][1].fill_(val)
        tensor_0 = torch.ones(shape, device="cpu")
        tensor_cpu = tensor_0[:][1].fill_(val)

        self.assertEqual(tensor_mps, tensor_cpu)

        shape = [1, 10]
        val = 0.0
        tensor = torch.ones(shape, device="mps")
        val_tensor_mps = torch.tensor(val, device="mps")
        tensor_mps = tensor[:, 9].fill_(val_tensor_mps)
        tensor_0 = torch.ones(shape, device="cpu")
        val_tensor_cpu = torch.tensor(val, device="cpu")
        tensor_cpu = tensor_0[:, 9].fill_(val_tensor_cpu)

        self.assertEqual(tensor_mps, tensor_cpu)

    def test_cdist_large(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(100, 10, device=device)
            y = torch.randn(100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_large_batch(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 100, 10, device=device)
            y = torch.randn(4, 3, 100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_non_contiguous(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    def test_cdist_non_contiguous_batch(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 2, 5, 7, device=device).mT
            y = torch.randn(4, 3, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 2, 7, 5, device=device)
            y = torch.randn(7, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(4, 5, 7, device=device).mT
            y = torch.randn(4, 3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    def test_cdist_euclidean_large(self, device="mps"):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    def test_cdist_same_inputs(self, device="mps"):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward passs does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()


    def _brute_cdist(self, x, y, p=2):
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    def test_cdist_norm(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 1, 1.5, 2.5, float('inf')]:
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    def test_cdist_norm_batch(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    def test_mm(self):
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.mm(B, C).cpu()
        torch.testing.assert_close(D, torch.full((5, 5), 6.0))

    def test_linalg_cross(self):
        def helper(dtype):
            device = "mps"
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
            else:
                x = torch.rand(100, 3, 100, dtype=dtype, device=device)
                y = torch.rand(100, 3, 100, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)

            # test for broadcastable inputs
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (1, 3, 2), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (4, 3, 1), dtype=dtype, device=device)
            else:
                x = torch.rand(1, 3, 2, dtype=dtype, device=device)
                y = torch.rand(4, 3, 1, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)
        [helper(dtype) for dtype in [torch.int32, torch.int64, torch.float32]]

    def test_cross(self):
        a = torch.randn(4, 3, device="mps")
        b = torch.randn(4, 3, device="mps")
        a_cpu = a.to("cpu")
        b_cpu = b.to("cpu")
        res = torch.cross(a, b, dim=1)
        res_cpu = torch.cross(a_cpu, b_cpu, dim=1)
        self.assertEqual(res, res_cpu)

    def test_addmm(self):
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.addmm(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 5), 7.0))

    def test_bmm(self):
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.bmm(batch1_cpu, batch2_cpu)
        output_mps = torch.bmm(batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_addr(self):
        A = torch.ones(5, 10).to("mps")
        B = torch.ones(5).to("mps")
        C = torch.ones(10).to("mps")
        D = torch.addr(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 10), 2.0))

    def test_trace(self):
        M_cpu = torch.randn(3, 3)
        M_mps = M_cpu.detach().clone().to("mps")

        output_cpu = torch.trace(M_cpu)
        output_mps = torch.trace(M_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_addbmm(self):
        M_cpu = torch.randn(3, 5)
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        M_mps = M_cpu.detach().clone().to("mps")
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.addbmm(M_cpu, batch1_cpu, batch2_cpu)
        output_mps = torch.addbmm(M_mps, batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_baddbmm(self):
        def helper(input_shape, batch1_shape, batch2_shape):
            M_cpu = torch.randn(input_shape)
            batch1_cpu = torch.randn(batch1_shape)
            batch2_cpu = torch.randn(batch2_shape)
            alpha = 1.2
            beta = 0.8

            M_mps = M_cpu.detach().clone().to("mps")
            batch1_mps = batch1_cpu.detach().clone().to("mps")
            batch2_mps = batch2_cpu.detach().clone().to("mps")

            output_cpu = torch.baddbmm(M_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
            output_mps = torch.baddbmm(M_mps, batch1_mps, batch2_mps, beta=beta, alpha=alpha)

            self.assertEqual(output_cpu, output_mps)
            self.assertEqual(output_cpu.size(), output_mps.size())

        helper(input_shape=(3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(10, 3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(1, 77, 77), batch1_shape=(8, 77, 64), batch2_shape=(8, 64, 77))

    def test_local_scalar_dense_mps(self):
        x_cpu = torch.randn(1)
        y_mps = x_cpu.to("mps")
        torch.testing.assert_close(x_cpu.item(), y_mps.item())

    def test_linear_1d_weight(self):
        device = 'cpu'
        projected = torch.rand([8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

        projected = torch.rand([1, 8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

    def test_linear_bias(self):
        def helper(bias_shape):
            device = "cpu"
            x = torch.randn(2, 2, 2, 64, device=device)
            linear = torch.nn.Linear(64, 4, device=device)
            linear.bias = torch.nn.Parameter(torch.randn(bias_shape, dtype=torch.float32, device=device))
            y = linear(x)
            device = "mps"
            x_mps = x.to(device)
            linear.to(device)
            y_mps = linear(x_mps)
            self.assertEqual(y, y_mps)

        helper(())
        helper((2, 4))

    def _linear_helper(self, in_features, out_features, shape, bias=True, backward_pass=False):
        cpu_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cpu", bias=bias)
        mps_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="mps", bias=bias)

        # Use the same weights and bias as the ones from the cpu
        mps_linear.weight.data = cpu_linear.weight.data.detach().clone().to("mps")

        if bias:
            mps_linear.bias.data = cpu_linear.bias.data.detach().clone().to("mps")

        linear_mps_input = torch.randn(shape).to('mps')
        linear_cpu_input = linear_mps_input.detach().clone().to('cpu')

        if backward_pass:
            linear_mps_input = linear_mps_input.requires_grad_()
            linear_cpu_input = linear_cpu_input.requires_grad_()

        linear_cpu_output = cpu_linear(linear_cpu_input)
        linear_mps_output = mps_linear(linear_mps_input)

        self.assertEqual(linear_cpu_output, linear_mps_output.to('cpu'))
        self.assertEqual(linear_cpu_output.size(), linear_mps_output.size())

        if backward_pass:
            cpu_grad = torch.rand_like(linear_cpu_output, requires_grad=True)
            grad = cpu_grad.detach().to('mps').requires_grad_()

            linear_cpu_output.backward(gradient=cpu_grad, create_graph=True)
            linear_mps_output.backward(gradient=grad, create_graph=True)

            self.assertEqual(linear_cpu_input.grad.size(), linear_mps_input.grad.size())
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            self.assertEqual(cpu_linear.weight.grad.size(), mps_linear.weight.grad.size())
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)
            if bias:
                self.assertEqual(cpu_linear.bias.grad.size(), mps_linear.bias.grad.size())
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            # test gradgrad
            x_grad_out = torch.rand_like(linear_cpu_input)
            x_grad_out_mps = x_grad_out.to("mps")
            w_grad_out = torch.rand_like(cpu_linear.weight)
            w_grad_out_mps = w_grad_out.to("mps")

            linear_cpu_input.grad.detach().zero_()
            linear_mps_input.grad.detach().zero_()
            cpu_linear.weight.grad.detach().zero_()
            mps_linear.weight.grad.detach().zero_()
            if bias:
                b_grad_out = torch.rand_like(cpu_linear.bias)
                b_grad_out_mps = b_grad_out.to("mps")
                cpu_linear.bias.grad.detach().zero_()
                mps_linear.bias.grad.detach().zero_()

            linear_cpu_input.grad.backward(x_grad_out, retain_graph=True)
            linear_mps_input.grad.backward(x_grad_out_mps, retain_graph=True)
            cpu_linear.weight.grad.backward(w_grad_out, retain_graph=True)
            mps_linear.weight.grad.backward(w_grad_out_mps, retain_graph=True)
            if bias:
                cpu_linear.bias.grad.backward(b_grad_out, retain_graph=True)
                mps_linear.bias.grad.backward(b_grad_out_mps, retain_graph=True)

            self.assertEqual(cpu_grad.grad, grad.grad)
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad)
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad)
            if bias:
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad)

    def test_linear1D(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=False)

    def test_linear1D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=True)

    def test_linear2D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=False)

    def test_linear2D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=True)

    def test_linear2D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=False)

    def test_linear2D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=True)

    def test_linear3D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_linear3D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

    def test_randperm(self, device="mps"):
        rng_device = None
        for n in (5, 100, 50000, 100000):
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                if n > 256 and dtype == torch.bfloat16:
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1.cpu().sort().values.long(), torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(res.cpu().sort().values.long(), torch.arange(n, device=device))

    # Test forward maxpool2d
    def test_max_pool2d(self):
        def helper(shape, ks, padding=0, dilation=1, ceil_mode=False, return_indices=False, test_ties=False):

            cpu_x = None
            if (test_ties):
                cpu_x = torch.ones(shape, device='cpu', dtype=torch.float, requires_grad=True)
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks, padding=padding, dilation=dilation,
                                      ceil_mode=ceil_mode, return_indices=return_indices)

            if (return_indices is False):
                y = pool(x)
                ref_y = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(x.grad, cpu_x.grad)
            else:
                y, idx = pool(x)
                ref_y, ref_idx = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(idx, ref_idx)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test with no batch dimension
        helper((8, 4, 4), ks=2)
        helper((2, 8, 4, 4), ks=2)
        helper((1, 1000, 32, 32), ks=4)
        helper((1, 1000, 1, 4), ks=(1, 4))  # test for max_pool1d
        # Test padding
        helper((1, 1000, 32, 32), ks=4, padding=1)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1))  # test for max_pool1d
        # Test dilation
        helper((1, 1000, 32, 32), ks=4, dilation=2)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2))  # test for max_pool1d
        # Test ceil mode
        helper((1, 1000, 32, 32), ks=4, ceil_mode=True)
        helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True)  # test for max_pool1d

        # Test return indices
        for test_ties in [False, True]:
            # Test with no batch dimension
            helper((8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((2, 8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 32, 32), ks=4, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test padding
            helper((1, 1000, 32, 32), ks=4, padding=1, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test dilation
            helper((1, 1000, 32, 32), ks=4, dilation=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test ceil mode
            helper((1, 1000, 32, 32), ks=4, ceil_mode=True, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True,
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d

    def test_adaptive_avg_pool2d_output_size_one(self):
        def helper(size, memory_format):
            x = torch.randint(1, 10, size, dtype=torch.float, device='mps', requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()    # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        helper((2, 3, 6, 6), torch.contiguous_format)

    def test_masked_scatter(self):
        def helper(shape):
            x_mps = torch.randn(shape, device="mps")
            x_cpu = x_mps.detach().clone().cpu()

            mask_mps = torch.rand(shape, device="mps") < 0.6
            mask_cpu = mask_mps.detach().clone().cpu()

            y_mps = torch.randn(shape, device="mps")
            y_cpu = y_mps.detach().clone().cpu()

            y_mps.masked_scatter_(mask_mps, x_mps)
            y_cpu.masked_scatter_(mask_cpu, x_cpu)

            self.assertEqual(y_mps, y_cpu)
        helper([2, 5])
        helper([10, 10])
        helper([5, 10, 3])
        helper([10, 5, 10, 3])
        helper([10, 5, 10, 3, 20])

    def test_masked_fill(self):
        device = "mps"
        dtype = torch.float32
        mask_dtype = torch.bool

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            num_dest = 10
            dst = torch.zeros(num_dest, dtype=dtype, device=device)
            mask = torch.randint(2, (num_dest,), dtype=mask_dtype, device=device)
            val = random.random()
            dst2 = torch.zeros(num_dest, dtype=dtype)
            mask_cpu = mask.to("cpu")

            dst.masked_fill_(mask, val)
            for i in range(num_dest):
                if mask_cpu[i]:
                    dst2[i] = val
            self.assertEqual(dst.to("cpu"), dst2, atol=0, rtol=0)

            # test non-contiguous case
            dst = ((torch.randn(num_dest, num_dest, num_dest) * 10).to(dtype)).permute((2, 0, 1))
            dst2 = dst.contiguous()
            if dtype.is_complex:
                mask = dst.abs() > 0
            else:
                mask = dst > 0
            self.assertTrue(not dst.is_contiguous())
            self.assertTrue(dst2.is_contiguous())
            dst.masked_fill_(mask.to(mask_dtype), val)
            dst2.masked_fill_(mask.to(mask_dtype), val)
            self.assertEqual(dst, dst2, atol=0, rtol=0)

            if mask_dtype == torch.uint8:
                self.assertEqual(len(w), 3)

                warn = 'masked_fill_ received a mask with dtype torch.uint8,'
                for wi in w:
                    self.assertEqual(str(wi.message)[0:52], str(warn))
            else:
                self.assertEqual(len(w), 0)

    def test_nhwc_operation(self):
        def helper(shape, channels_last=False):
            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # This passes
            self.assertEqual(x, cpu_x)

        helper((2, 2, 2, 2), True)

    # Test forward batch norm
    def test_batch_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, training=False, channels_last=False,
                   track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                y = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                   weight=weight,
                                                   bias=bias,
                                                   training=training,
                                                   momentum=momentum, eps=eps)
                ref_y = torch.nn.functional.batch_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                       weight=cpu_weight,
                                                       bias=cpu_bias,
                                                       training=training,
                                                       momentum=momentum, eps=eps)

            else:

                batchnorm_op = None
                mps_batchnorm_op = None

                if (len(shape) == 3):
                    batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 4):
                    batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 5):
                    batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')

                if (track_running_stats):
                    batchnorm_op.running_mean = cpu_running_mean
                    batchnorm_op.running_var = cpu_running_var
                    mps_batchnorm_op.running_mean = running_mean
                    mps_batchnorm_op.running_var = running_var
                if (wts):
                    batchnorm_op.weight = torch.nn.Parameter(cpu_weight)
                    batchnorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_batchnorm_op.weight = torch.nn.Parameter(weight)
                    mps_batchnorm_op.bias = torch.nn.Parameter(bias)

                ref_y = batchnorm_op(cpu_x)
                y = mps_batchnorm_op(x)

            self.assertEqual(y, ref_y)
            if (not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_batchnorm_op.running_mean, batchnorm_op.running_mean)
                self.assertEqual(mps_batchnorm_op.running_var, batchnorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if (wts):
                if (not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_batchnorm_op.weight.grad, batchnorm_op.weight.grad)
                    self.assertEqual(mps_batchnorm_op.bias.grad, batchnorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if (channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if (track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_batch_norm_backward(self):
        inputs = torch.rand(1, 8, 4, 4, device="mps", requires_grad=True)
        x = torch.nn.BatchNorm2d(8).to("mps")
        y = torch.nn.BatchNorm2d(8).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_layer_norm_backward(self):
        inputs = torch.rand(4, 4, device="mps", requires_grad=True)
        x = torch.nn.LayerNorm(4).to("mps")
        y = torch.nn.LayerNorm(4).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_norm(self):
        a = torch.arange(9, dtype=torch.float, device="mps") - 4
        b = a.reshape((3, 3))

        a_cpu = torch.arange(9, dtype=torch.float, device="cpu") - 4
        b_cpu = a_cpu.reshape((3, 3))

        res = torch.norm(a)
        res_cpu = torch.norm(a_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(b)
        res_cpu = torch.norm(b_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(a, float('inf'))
        res_cpu = torch.norm(a_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        res = torch.norm(b, float('inf'))
        res_cpu = torch.norm(b_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float, device="mps")
        c_cpu = torch.tensor([[1, 2, 3], [-1, 1, 4]] , dtype=torch.float, device="cpu")

        res = torch.norm(c, dim=0)
        res_cpu = torch.norm(c_cpu, dim=0)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, dim=1)
        res_cpu = torch.norm(c_cpu, dim=1)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, p=1, dim=1)
        res_cpu = torch.norm(c_cpu, p=1, dim=1)
        self.assertEqual(res, res_cpu)

        d = torch.arange(8, dtype=torch.float, device="mps").reshape(2, 2, 2)
        d_cpu = torch.arange(8, dtype=torch.float, device="cpu").reshape(2, 2, 2)

        res = torch.norm(d, dim=(1, 2))
        res_cpu = torch.norm(d_cpu, dim=(1, 2))
        self.assertEqual(res, res_cpu)

        res = torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        res_cpu = torch.norm(d_cpu[0, :, :]), torch.norm(d_cpu[1, :, :])
        self.assertEqual(res, res_cpu)

    def test_linalg_vector_norm(self):
        x_mps = torch.tensor([0, 0, 0, 2, 3], dtype=torch.float, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        res_mps = torch.linalg.vector_norm(x_mps, ord=0)
        res_cpu = torch.linalg.vector_norm(x_cpu, ord=0)
        self.assertEqual(res_mps, res_cpu)

        a_mps = torch.arange(27, dtype=torch.float, device="mps") - 4
        a_cpu = torch.arange(27, dtype=torch.float, device="cpu") - 4

        B_mps = a_mps.reshape(3, 3, 3)
        B_cpu = a_cpu.reshape(3, 3, 3)

        res_mps = torch.linalg.vector_norm(a_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(a_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        res_mps = torch.linalg.vector_norm(B_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        for dim in range(0, B_mps.dim()):
            res_mps = torch.linalg.vector_norm(B_mps, ord=3.5, dim=dim)
            res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5, dim=dim)
            self.assertEqual(res_mps, res_cpu)


    def test_layer_norm(self):
        # TODO: Test non-contiguous
        def helper(input_shape, normalized_shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='cpu', dtype=dtype)
            mps_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='mps', dtype=dtype)
            cpu_wt = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()
            cpu_bias = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            if (elementwise_affine):
                cpu_op.weight = torch.nn.Parameter(cpu_wt)
                mps_op.weight = torch.nn.Parameter(wt)
                cpu_op.bias = torch.nn.Parameter(cpu_bias)
                mps_op.bias = torch.nn.Parameter(bias)

            cpu_result = cpu_op(cpu_x)
            result = mps_op(x)

            cpu_grad = torch.randn(cpu_result.shape)
            grad = cpu_grad.to('mps')

            cpu_result.backward(cpu_grad)
            result.backward(grad)

            self.assertEqual(result, cpu_result)
            self.assertEqual(x.grad, cpu_x.grad)
            if (elementwise_affine):
                self.assertEqual(mps_op.weight.grad, cpu_op.weight.grad)
                self.assertEqual(mps_op.bias.grad, cpu_op.bias.grad)

        for elementwise_affine in [True, False]:
            helper((2, 2, 2, 2), (2, 2), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5), (4, 5), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5, 6), (4, 5, 6), elementwise_affine=elementwise_affine)

        # Regression test for https://github.com/pytorch/pytorch/issues/96113
        torch.nn.LayerNorm((16,), elementwise_affine=True).to("mps")(torch.randn(1, 2, 16).to("mps", dtype=torch.float16))

    def test_instance_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, channels_last=False, track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                ref_y = torch.nn.functional.instance_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                          weight=cpu_weight,
                                                          bias=cpu_bias,
                                                          momentum=momentum, eps=eps)
                y = torch.nn.functional.instance_norm(x, running_mean, running_var,
                                                      weight=weight,
                                                      bias=bias,
                                                      momentum=momentum, eps=eps)

            else:

                instancenorm_op = None
                mps_instancenorm_op = None

                if (len(shape) == 3):
                    instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif (len(shape) == 4):
                    instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif (len(shape) == 5):
                    instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')

                if (track_running_stats):
                    instancenorm_op.running_mean = cpu_running_mean
                    instancenorm_op.running_var = cpu_running_var
                    mps_instancenorm_op.running_mean = running_mean
                    mps_instancenorm_op.running_var = running_var
                if (wts):
                    instancenorm_op.weight = torch.nn.Parameter(cpu_weight)
                    instancenorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_instancenorm_op.weight = torch.nn.Parameter(weight)
                    mps_instancenorm_op.bias = torch.nn.Parameter(bias)

                ref_y = instancenorm_op(cpu_x)
                y = mps_instancenorm_op(x)

            self.assertEqual(y, ref_y)
            if (not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_instancenorm_op.running_mean, instancenorm_op.running_mean)
                self.assertEqual(mps_instancenorm_op.running_var, instancenorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if (wts):
                if (not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_instancenorm_op.weight.grad, instancenorm_op.weight.grad)
                    self.assertEqual(mps_instancenorm_op.bias.grad, instancenorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if (channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if (track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_weight_norm(self):
        def helper(dim, layer='linear', dtype=torch.float32):
            # linear layer
            if layer == 'linear':
                cpu_x = torch.randn((2, 5), device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

                cpu_weight = torch.randn(10, 5, device='cpu', dtype=dtype, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()

                cpu_bias = torch.randn(10, device='cpu', dtype=dtype, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

                cpu_linear = torch.nn.Linear(5, 10, device='cpu')
                linear = torch.nn.Linear(5, 10, device='mps')

                with torch.no_grad():
                    cpu_linear.weight.copy_(cpu_weight)
                    cpu_linear.bias.copy_(cpu_bias)
                    linear.weight.copy_(weight)
                    linear.bias.copy_(bias)

                cpu_norm = torch.nn.utils.weight_norm(cpu_linear, dim=dim)
                norm = torch.nn.utils.weight_norm(linear, dim=dim)

                cpu_out = cpu_norm(cpu_x)
                out = norm(x)

                self.assertEqual(cpu_out, out)

                cpu_grad = torch.randn(cpu_out.shape)
                grad = cpu_grad.to('mps')
                cpu_out.backward(gradient=cpu_grad)
                out.backward(gradient=grad)

                self.assertEqual(cpu_linear.weight_g.grad, linear.weight_g.grad)
                self.assertEqual(cpu_linear.weight_v.grad, linear.weight_v.grad)

                self.assertEqual(x.grad, cpu_x.grad)

            # conv layer
            if layer == 'conv':
                cpu_x = torch.randn((3, 5, 5), device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

                cpu_conv = torch.nn.Conv2d(3, 3, 3, device='cpu')
                conv = torch.nn.Conv2d(3, 3, 3, device='mps')

                with torch.no_grad():
                    conv.weight.copy_(cpu_conv.weight)
                    conv.bias.copy_(cpu_conv.bias)

                cpu_norm = torch.nn.utils.weight_norm(cpu_conv, dim=dim)
                norm = torch.nn.utils.weight_norm(conv, dim=dim)

                cpu_out = cpu_conv(cpu_x)
                out = conv(x)

                self.assertEqual(cpu_out, out)

                cpu_grad = torch.randn(cpu_out.shape)
                grad = cpu_grad.to('mps')
                cpu_out.backward(gradient=cpu_grad)
                out.backward(gradient=grad)

                self.assertEqual(cpu_conv.weight_g.grad, conv.weight_g.grad)
                self.assertEqual(cpu_conv.weight_v.grad, conv.weight_v.grad)

                self.assertEqual(x.grad, cpu_x.grad)

        helper(0, layer='linear')
        helper(1, layer='linear')
        helper(-1, layer='linear')

        helper(0, layer='conv')
        helper(1, layer='conv')
        helper(2, layer='conv')
        helper(3, layer='conv')
        helper(-1, layer='conv')

    # Test conv2d
    def test_conv2d_unit(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if (bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv2d(x, wt, bias=bias, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups)
            ref_y = torch.nn.functional.conv2d(cpu_x, cpu_wt, bias=cpu_bias, stride=stride,
                                               padding=padding, dilation=dilation, groups=groups)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)
            if (bias_shape is not None):
                self.assertEqual(bias.grad, cpu_bias.grad, atol=8e-04, rtol=10.4e-05)

        N = 1
        C_in = 3
        C_out = 64
        H = 64
        W = 64
        kH = 4
        kW = 4
        stride = 2
        padding = 1

        helper((N, C_in, H, W), (C_out, C_in, kH, kW), stride=stride, padding=padding)

        N = 4
        C_in = 16
        H = 32
        W = 32

        C_out = 8
        kH = 3
        kW = 3

        for groups in [1, 2, 4]:
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)

            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)

    # Test conv transpose 2d
    def test_conv_transpose2d(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   output_padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if (bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv_transpose2d(
                x, wt, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
            ref_y = torch.nn.functional.conv_transpose2d(
                cpu_x, cpu_wt, bias=cpu_bias, stride=stride, padding=padding,
                output_padding=output_padding, groups=groups, dilation=dilation)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)

            # if (bias_shape is not None):
            #  print(cpu_bias.grad)
            #  print(bias.grad.to('cpu'))
            #  self.assertEqual(bias.grad, cpu_bias.grad)

        N = 4
        C_in = 2
        H = 32
        W = 32

        C_out = 8
        groups = 1
        kH = 3
        kW = 3

        for stride in [1, 2, 3]:
            for padding in [0, 1, 2]:
                for output_padding in [0, 1, 2]:
                    for dilation in [1, 2]:
                        if (output_padding >= stride or output_padding >= dilation):
                            continue
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

    # Test sigmoid
    def test_sigmoid(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sigmoid_op = torch.nn.Sigmoid()

            y = sigmoid_op(x)
            ref_y = sigmoid_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    # Test tanh
    def test_tanh(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            tanh_op = torch.nn.Tanh()

            y = tanh_op(x)
            ref_y = tanh_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    def test_threshold(self):
        def helper(threshold, value, num_elems, inplace=False, requires_grad=True):
            m = nn.Threshold(threshold=threshold, value=value, inplace=inplace)

            input_cpu = torch.randn(num_elems, requires_grad=requires_grad, dtype=torch.float)
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            cpu_grad = torch.ones_like(output_cpu)
            mps_grad = cpu_grad.to('mps')

            self.assertEqual(output_cpu, output_mps)

            if requires_grad:
                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                self.assertEqual(input_cpu.grad, input_mps.grad)

        helper(threshold=0.1, value=20, num_elems=2)
        helper(threshold=-0.1, value=10, num_elems=10)
        helper(threshold=0.5, value=-15, num_elems=100)
        helper(threshold=1, value=10, num_elems=100, inplace=True, requires_grad=False)

    # Test pow
    def test_pow(self):
        def helper(shape):
            # aten::pow.Tensor_Tensor
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')
            z = torch.pow(x, y)
            ref_z = torch.pow(cpu_x, cpu_y)

            self.assertEqual(z, ref_z)

            # aten::pow.Tensor_Scalar
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            exp = random.random()
            z = torch.pow(x, exp)
            ref_z = torch.pow(cpu_x, exp)

            self.assertEqual(z, ref_z)

            # aten::pow.Scalar
            x = random.random()
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')
            z = torch.pow(x, y)
            ref_z = torch.pow(x, cpu_y)

            self.assertEqual(z, ref_z)

        helper((2, 8, 4, 5))

    # Test addcmul
    def test_addcmul(self):
        def helper(shape, value, xtype=torch.float32, ytype=None, ztype=None):
            def rand_helper(dtype):
                if dtype.is_floating_point:
                    return torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                return torch.randint(10, shape, dtype=dtype, device='cpu', requires_grad=False)

            cpu_x = rand_helper(xtype)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = rand_helper(ytype if ytype is not None else xtype)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = rand_helper(ztype if ztype is not None else xtype)
            z = cpu_z.detach().clone().to('mps')

            y = torch.addcmul(x, y, z, value=value)
            ref_y = torch.addcmul(cpu_x, cpu_y, cpu_z, value=value)

            self.assertEqual(y, ref_y)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.1)
        helper((2, 3, 4, 5), 0.2)
        helper((2, 8, 4, 5), 0.2)
        # Integral types
        helper((2, 2), 1.0, xtype=torch.int32)
        helper((2, 2), 2.0, xtype=torch.int16)

        # Mixed types
        helper((2, 2), 1.0, xtype=torch.float16, ytype=torch.float32)
        helper((3, 2), 1.0, ytype=torch.float16)
        helper((2, 3), 1.0, ztype=torch.float16)
        helper((2, 2), 1.0, xtype=torch.int32, ytype=torch.int16, ztype=torch.uint8)
        helper((2, 2), 1.0, ytype=torch.int16, ztype=torch.uint8)

    # Test addcdiv
    def test_addcdiv(self):
        def helper(shape, value):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # clamp to avoid division by 0
            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False).clamp_min_(0.1)
            cpu_out = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)

            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            mps_z = cpu_z.detach().clone().to('mps')
            mps_out = cpu_out.detach().clone().to('mps')

            result_div_mps = torch.addcdiv(mps_x, mps_y, mps_z, value=value)
            result_div_cpu = torch.addcdiv(cpu_x, cpu_y, cpu_z, value=value)
            self.assertEqual(result_div_mps, result_div_cpu)
            # test .out variant
            self.assertEqual(torch.addcdiv(mps_x, mps_y, mps_z, out=mps_out, value=value), result_div_cpu)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.2)
        helper((2, 3, 4, 5), 1.0)  # value of 1 should be ignored internally

    def test_buffer_size_match(self):
        # this test shouldn't cause any crash
        size = 16
        cpu_A = torch.rand(size, device='cpu')
        cpu_F = torch.rand(size, size, size, device='cpu')

        mps_A = cpu_A.to('mps')
        mps_F = cpu_F.to('mps')
        self.assertEqual(cpu_A @ cpu_F, mps_A @ mps_F)

    def test_transpose_inplace(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_x.transpose_(0, 1)
        mps_x.transpose_(0, 1)
        self.assertEqual(cpu_x, mps_x.to('cpu'))

    def test_expand_cpu_to_mps_copy(self):
        # https://github.com/pytorch/pytorch/issues/78642

        x = torch.tensor(1).expand([10]).to("mps")
        x_cpu = torch.tensor(1).expand([10])

        self.assertEqual(x_cpu, x.cpu())

    def test_cpu_to_strided_mps_copy(self):
        # https://github.com/pytorch/pytorch/issues/86975

        a1 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        b1 = torch.Tensor([-1, -1])
        a1[1:, 1] = b1

        a2 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        b2 = torch.Tensor([-1, -1]).to(torch.device("mps"))
        a2[1:, 1] = b2

        self.assertEqual(a1, a2)

    def test_view_slice_reshape(self):
        x = torch.randn([1, 4, 4], device="mps")
        y = x[0, :1, 1:]

        x_cpu = x.to("cpu")
        y_cpu = x_cpu[0, :1, 1:]

        r = y + 1
        r_cpu = y_cpu + 1
        self.assertEqual(r, r_cpu)

    def test_slice_reshape(self):
        x = torch.randn([1, 6, 4, 2], dtype=torch.float, device="mps")
        x_cpu = x.detach().clone().to("cpu")

        x = x[:, 3:].view(2, 3, 4, 1)
        x_cpu = x_cpu[:, 3:].view(2, 3, 4, 1)
        self.assertEqual(x, x_cpu)

        x = x + 2
        x_cpu = x_cpu + 2
        self.assertEqual(x, x_cpu)

    def test_reshape_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/95883
        B = 4
        T = 1

        lin_cpu = nn.Linear(10, 256)
        lin_mps = nn.Linear(10, 256, device="mps")

        # Use the same weights and bias as the ones from the cpu
        lin_mps.weight.data = lin_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        lin_mps.bias.data = lin_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        x_mps = torch.rand([B, T, 10], device="mps", requires_grad=True)
        x_cpu = x_mps.detach().clone().cpu().requires_grad_()
        x_mps = lin_mps(x_mps)
        x_cpu = lin_cpu(x_cpu)

        self.assertEqual(x_mps.shape, (B, T, 256))
        self.assertEqual(x_cpu.shape, (B, T, 256))

        cls_token_mps = torch.rand([1, 256], device="mps", requires_grad=True).repeat(B, 1, 1)
        cls_token_cpu = cls_token_mps.detach().clone().cpu()
        x_mps = torch.cat([cls_token_mps, x_mps], dim=1)
        x_cpu = torch.cat([cls_token_cpu, x_cpu], dim=1)

        x_mps = x_mps.transpose(0, 1)
        x_cpu = x_cpu.transpose(0, 1)

        target_mps = torch.rand_like(x_mps)
        target_cpu = target_mps.detach().clone().cpu()
        loss_mps = F.mse_loss(x_mps, target_mps)
        loss_cpu = F.mse_loss(x_cpu, target_cpu)
        self.assertEqual(loss_mps, loss_cpu)

        loss_mps.backward()
        loss_cpu.backward()
        self.assertEqual(x_mps.grad, x_cpu.grad)

    def test_stack(self):
        # https://github.com/pytorch/pytorch/issues/87856
        x_cpu = torch.tensor([[1, 2]])
        x_mps = x_cpu.detach().clone().to("mps")

        y_cpu = torch.stack((x_cpu[:, :1], x_cpu[:, -1:]), dim=-1)
        y_mps = torch.stack((x_mps[:, :1], x_mps[:, -1:]), dim=-1)

        self.assertEqual(y_cpu, y_mps)

        t_mps = torch.tensor([1, 2, 3, 4], device="mps")
        t_cpu = t_mps.detach().cpu().detach()

        x_mps = t_mps[2:]
        y_mps = t_mps[:2]

        x_cpu = t_cpu[2:]
        y_cpu = t_cpu[:2]

        res_mps = torch.stack((y_mps, x_mps), dim=-1)
        res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

        self.assertEqual(res_mps, res_cpu)

    def test_unsafe_chunk(self):
        # https://github.com/pytorch/pytorch/issues/91065
        a = torch.rand(5, dtype=torch.float32, device="cpu")
        ret = a.unsafe_chunk(4, 0)
        y = ret[0] * ret[2]
        a_mps = a.to("mps")
        ret_mps = a_mps.unsafe_chunk(4, 0)
        y_mps = ret_mps[0] * ret_mps[2]
        self.assertEqual(y, y_mps)

    def test_slice_casting(self):
        # generate random binary numbers
        cpu_in = torch.bernoulli(torch.empty(1, 1, 128, 128).uniform_(0, 1)).to(torch.uint8)
        mps_in = cpu_in.detach().clone().to("mps")
        # check copy_cast(unit8 -> bool) on tensors with storage offset
        cpu_out = cpu_in[:, :, 11 : 12, :12].to(torch.bool)
        mps_out = mps_in[:, :, 11 : 12, :12].to(torch.bool)
        self.assertEqual(cpu_out, mps_out)

    def test_slice_reshape_contg_view(self):
        import torch

        x_mps = torch.randn(1, 4800, 2, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        r_mps = x_mps + 2
        r_cpu = x_cpu + 2

        self.assertEqual(r_mps, r_cpu)

    def test_contiguous_slice_2d(self):
        def helper(shape):
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    t_mps = torch.randn(shape, device="mps")
                    t_cpu = t_mps.detach().clone().cpu()

                    y_mps = t_mps[i:, :j]
                    y_cpu = t_cpu[i:, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i:, j]
                    y_cpu = t_cpu[i:, j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i, :j]
                    y_cpu = t_cpu[i, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, :j]
                    y_cpu = t_cpu[:i, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j]
                    y_cpu = t_cpu[:i, j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j:]
                    y_cpu = t_cpu[:i, j:]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

        l = []
        for N in range(1, 3):
            l.append(N)
            for C in range(1, 3):
                l.append(C)
                helper(l)
                for D in range(1, 3):
                    l.append(D)
                    helper(l)
                    for H in range(1, 3):
                        l.append(H)
                        helper(l)
                        for W in range(1, 3):
                            l.append(W)
                            helper(l)
                            l.pop()
                        l.pop()
                    l.pop()
                l.pop()
            l.pop()

        helper([9, 15, 4])
        helper([9, 3, 2])
        helper([3, 4, 18, 22])
        helper([3, 4, 18, 22, 150])

    def test_contiguous_slice_3d(self):
        x = torch.randn(2, 3, 3, device="mps")
        x_cpu = x.detach().clone().cpu()
        x = x[:1]
        x_cpu = x_cpu[:1]
        out = x[:, 0:1, 0:1] * x[:, 1:2, 1:2]
        out_cpu = x_cpu[:, 0:1, 0:1] * x_cpu[:, 1:2, 1:2]
        self.assertEqual(out, out_cpu)

    def test_view_slice(self):
        # https://github.com/pytorch/pytorch/issues/83995
        NUM_SAMPLES = 60
        s = (0, 1)

        X = torch.rand(8000, 3, dtype=torch.float32, device='cpu')
        X_mps = X.detach().clone().to("cpu")

        idx = torch.randint(0, X.shape[0], (1,)).repeat(len(s))
        pts = torch.randint(0, X.shape[0], (NUM_SAMPLES, X.shape[1]))
        idx_mps = idx.to("mps")
        pts_mps = pts.to("mps")
        pts[:, s] = idx
        pts_mps[:, s] = idx_mps

        actual_pts = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float)
        actual_pts_mps = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float, device="mps")

        for i in range(NUM_SAMPLES):
            for j in range(X.shape[1]):
                actual_pts_mps[i, j] = X_mps[pts_mps[i, j], j]
                actual_pts[i, j] = X[pts[i, j], j]
                self.assertEqual(actual_pts[i, j], actual_pts_mps[i, j])

    def test_slice_scatter(self):
        shape = (4, 4)
        tensor = torch.randint(10, shape, device="mps")
        tensor_before = tensor.clone()
        torch.empty(shape[0], shape[1] * 2, device="mps")[:, ::2].copy_(tensor)
        torch.testing.assert_close(tensor, tensor_before)

    def test_slice(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = (torch.tensor(values, device='mps', dtype=torch.float))

        cpu_slice1 = cpu_x[:2, :]
        mps_slice1 = mps_x[:2, :]
        self.assertEqual(cpu_slice1, mps_slice1)

        cpu_slice2 = cpu_x[:, :1]
        mps_slice2 = mps_x[:, :1]
        self.assertEqual(cpu_slice2, mps_slice2)

        cpu_slice3 = cpu_x[1:2, :]
        mps_slice3 = mps_x[1:2, :]
        self.assertEqual(cpu_slice3, mps_slice3.to('cpu'))

        cpu_slice4 = cpu_x[1, :]
        mps_slice4 = mps_x[1, :].to('cpu')
        self.assertEqual(cpu_slice4, mps_slice4)

    def test_scalar_from_slice_unary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        tensor_list = torch.tensor([1.0, 1.2], device="mps")

        for scalar in tensor_list:
            r_mps = torch.ceil(scalar)
            r_cpu = torch.ceil(scalar.to("cpu"))
            self.assertEqual(r_mps.cpu(), r_cpu)

    def test_scalar_from_slice_binary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        def helper(binary_op):
            tensor_list = torch.tensor([1.0, 1.2, 2.5, 1.0], device="mps")

            for scalar in tensor_list:
                r_mps = binary_op(scalar, 1.0)
                r_cpu = binary_op(scalar.cpu(), 1.0)
                self.assertEqual(r_mps.cpu(), r_cpu)
        helper(torch.sub)
        helper(torch.add)
        helper(torch.not_equal)
        helper(torch.eq)

    def test_slice_contiguous_view(self):
        # https://github.com/pytorch/pytorch/issues/77750

        def helper(operator):
            t_mps = torch.tensor([1, 2, 3, 4], device="mps")
            t_cpu = torch.tensor([1, 2, 3, 4], device="cpu")

            # contiguous view
            x_mps = t_mps[2:]  # 3, 4
            y_mps = t_mps[:2]  # 1, 2

            x_cpu = t_cpu[2:]
            y_cpu = t_cpu[:2]

            res_mps = res_cpu = None
            if operator == "<=":
                res_mps = x_mps <= y_mps
                res_cpu = x_cpu <= y_cpu
            elif operator == "<":
                res_mps = x_mps < y_mps
                res_cpu = x_cpu < y_cpu
            elif operator == ">=":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == ">":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == "==":
                res_mps = x_mps == y_mps
                res_cpu = x_cpu == y_cpu
            elif operator == "!=":
                res_mps = x_mps != y_mps
                res_cpu = x_cpu != y_cpu
            elif operator == "stack":
                res_mps = torch.stack((y_mps, x_mps), dim=-1)
                res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

            self.assertEqual(res_mps, res_cpu)

        for op in ["<=", "<", ">=", ">", "==", "!=", "stack"]:
            helper(op)

    def test_slice_of_slice(self):
        x = torch.tensor([0.5, 0.5], device="cpu")
        x_mps = torch.tensor([0.5, 0.5], device="mps")

        tensor = x[1][None]
        tensor_mps = x_mps[1][None]

        res = tensor.ne(0)
        res_mps = tensor_mps.ne(0)

        self.assertEqual(res, res_mps)

    def test_index_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/78107

        a = torch.tensor([8.2670e-01, -1.0293e+00])
        b_cpu = a[0]
        c_cpu = a[1]

        # both 'b' and 'c' are views of 'a'
        # 'b' has a storage offset of 0, while 'c' has a storage offset of 1
        # when copying from 'cpu' to 'mps', c will have a storage_offset of 1 which needs to be taking into account,
        # otherwise it ends with same value as 'b'
        b = b_cpu.to('mps')
        c = c_cpu.to('mps')

        res_mps = b > c
        res_cpu = b_cpu > c_cpu
        self.assertEqual(res_mps, res_cpu)

        res_mps = c > b
        res_cpu = c_cpu > b_cpu
        self.assertEqual(res_mps, res_cpu)

    def test_flatten(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_flatten1 = cpu_x.flatten()
        mps_flatten1 = mps_x.flatten().to('cpu')
        self.assertEqual(cpu_flatten1, mps_flatten1)

        cpu_flatten2 = cpu_x.flatten(start_dim=1)
        mps_flatten2 = mps_x.flatten(start_dim=1).to('cpu')
        self.assertEqual(cpu_flatten2, mps_flatten2)

        cpu_flatten3 = cpu_x.flatten(end_dim=1)
        mps_flatten3 = mps_x.flatten(end_dim=1).to('cpu')
        self.assertEqual(cpu_flatten3, mps_flatten3)

    # Test repeat
    def test_repeat(self):
        def helper(shape, repeats):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = x.repeat(repeats)
            ref_y = cpu_x.repeat(repeats)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5), (2, 3, 4, 5))
        helper((2, 3, 4), (4, 3, 2, 5, 7, 2))
        helper((3, 4, 5), (2, 3, 4, 5))
        helper((3, 4, 5), (2, 2, 2))

    def test_torch_repeat_interleave(self, device="mps"):
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # exercise single argument function signature
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        for dtype in [torch.int, torch.long]:
            lengths = torch.tensor([1, 2], dtype=dtype, device="mps")
            output_size = torch.sum(lengths)
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            self.assertEqual(a.dtype, y.dtype)
            self.assertEqual(a.size(), torch.Size([3, 2]))

            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            self.assertEqual(a_with_output.dtype, y.dtype)
            self.assertEqual(a_with_output.size(), torch.Size([3, 2]))

    def test_repeat_interleave(self, device="mps"):
        x = torch.tensor([0, 1, 2, 3], device=device)
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        # Prior to macos 13.3, input of dtype=torch.int64 returns dtype=torch.int32
        self.assertEqual(torch.repeat_interleave(x), expected, exact_dtype=product_version >= 13.3)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        y = torch.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0), device=device)
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = torch.tensor([], dtype=torch.int64, device=device)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    def test_repeat_interleave_simple(self):
        def helper(shape, dtype=torch.float32, num_repeats=torch.Tensor(), dim=None):
            x = torch.randn(shape, dtype=dtype, device="mps")
            x_cpu = x.detach().clone().cpu()

            num_repeats_cpu = num_repeats.detach().clone().cpu()

            repeats = torch.repeat_interleave(x, num_repeats, dim)
            repeats_cpu = torch.repeat_interleave(x_cpu, num_repeats_cpu, dim)

            self.assertEqual(repeats, repeats_cpu)
        helper(shape=3, num_repeats=torch.tensor([100], device="mps"))
        helper(shape=(2, 2), num_repeats=torch.tensor([3, 3], device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.arange(10, device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.randint(0, 100, (15, ), device="mps"), dim=1)
        helper(shape=(10, 15, 30), num_repeats=torch.randint(0, 100, (30, ), device="mps"), dim=2)

    def test_count_nonzero(self):
        def helper(dtype):
            n = [
                [[1, 0, 2], [3, 0, 2], [7, 9, -4]],
                [[0, 2, 3], [3, 2, 1], [2, 0, 0]],
            ]
            cpu_x = torch.tensor(n, dtype=dtype)
            mps_x = torch.tensor(n, dtype=dtype).to('mps')

            # All non-zeros
            self.assertEqual(
                torch.count_nonzero(cpu_x),
                torch.count_nonzero(mps_x)
            )

            # dim=1
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=1),
                torch.count_nonzero(mps_x, dim=1)
            )

            # dim=(0, 1)
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=(0, 1)),
                torch.count_nonzero(mps_x, dim=(0, 1))
            )
        helper(torch.int32)
        helper(torch.int64)
        helper(torch.float16)
        helper(torch.float32)

    def _test_module_empty_input(self, module, inp, check_size=True):
        inp.requires_grad_(True)
        out = module(inp)
        gO = torch.rand_like(out)
        out.backward(gO)
        if check_size:
            self.assertEqual(out.size(), inp.size())
        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))
        self.assertEqual(inp.grad, torch.zeros_like(inp))

    # Test dtype casting, with and without simultaneous device change
    def test_to(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        self.assertEqual(cpu_x.int(), mps_x.int().cpu())
        self.assertEqual(cpu_x.bool(), mps_x.bool().cpu())
        self.assertEqual(cpu_x.float(), mps_x.float().cpu())

        self.assertEqual(torch.tensor(1.3, device='mps').int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.0, device='mps').bool().cpu(), torch.tensor(False))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().cpu(), torch.tensor(True))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().float().cpu(),
                         torch.tensor(1.0))
        self.assertEqual(torch.tensor(4.25, device='mps').to('cpu', torch.int),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(4.25, device='cpu').to('mps', torch.int).cpu(),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(-8.34, device='cpu').to('mps', torch.int),
                         torch.tensor(-8.34, device='cpu').to('mps').to(torch.int))
        # Cast int8 and uint8 to float and compare results
        # See https://github.com/pytorch/pytorch/issues/80009 for more details
        cpu_byte = torch.tensor([60, 160, 20, 220], dtype=torch.uint8)
        cpu_char = torch.tensor([60, -60, 20, -120], dtype=torch.uint8)
        for x_cpu in [cpu_byte, cpu_char]:
            x_mps = x_cpu.to('mps')
            self.assertEqual(x_mps.to(torch.float32), x_cpu.to(torch.float32))


    def test_setitem_scalar(self) -> None:
        device = 'mps'
        for dtype in [torch.int32, torch.float32, torch.int64]:
            for i in range(3, 6):
                for j in range(3, 6):
                    t = torch.zeros(i, j, dtype=dtype, device=device)
                    self.assertEqual(t.sum(), 0)
                    t[1, 1] = 1
                    t[2, 1] = j
                    t[1, 2] = i
                    self.assertEqual(t[1, 1], 1)
                    self.assertEqual(t[1, 2], i)
                    self.assertEqual(t[2, 1], j)
                    self.assertEqual(t.sum(), 1 + i + j)

    def test_stride_of_strides(self) -> None:
        x = torch.rand(32, 1, device='mps')
        y = x.as_strided(size=(32, 2), stride=(1, 0))
        # Casting stride of strided tensor to CPU use to crash with "buffer is not large enough." assert
        # See https://github.com/pytorch/pytorch/issues/79181#issuecomment-1154683435
        z = y.as_strided(size=(32, 3), stride=(1, 0)).to("cpu")
        self.assertEqual(x.to("cpu").as_strided(size=(32, 3), stride=(1, 0)), z)

    def test_type_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.type(to_dtype)
            res_mps = a_mps.type(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.LongTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.FloatTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.IntTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ShortTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.HalfTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.CharTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ByteTensor)

    def test_to_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.to(to_dtype)
            res_mps = a_mps.to(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.int64)
        helper([9.0, 3.0, 5.0, 4.0], torch.float)
        helper([9.0, 3.0, 5.0, 4.0], torch.int32)
        helper([9.0, 3.0, 5.0, 4.0], torch.short)
        helper([9.0, 3.0, 5.0, 4.0], torch.half)
        helper([9.0, 3.0, 5.0, 4.0], torch.int8)
        helper([9.0, 3.0, 5.0, 4.0], torch.uint8)

    def test_storage_offset_greater_than_src_nbytes(self):
        # https://github.com/pytorch/pytorch/issues/80844
        n_tensors = 100
        n_tensor_elems = 784
        elems = torch.arange(n_tensors * n_tensor_elems, dtype=torch.float32)

        tensor_list = []
        for i in range(0, n_tensors - 1):
            # create a list of contiguous view tensors (view tensor created by the slice op)
            t = elems[n_tensor_elems * i : n_tensor_elems * (i + 1)]
            tensor_list.append(t)

        for i in range(0, n_tensors - 1):
            t = tensor_list[i].view(1, n_tensor_elems)
            t_mps = t.to("mps")
            self.assertEqual(t, t_mps.cpu(), f"i={i}")

    # See https://github.com/pytorch/pytorch/issues/82427
    # and https://github.com/pytorch/pytorch/issues/83692
    def test_full_bugs(self):
        # Test should not crash
        x = torch.full((3, 3), True, device='mps')
        # torch.full should work for uint8
        y_mps = torch.full((2, 2), 247, device='mps', dtype=torch.uint8)
        y_cpu = torch.full((2, 2), 247, device='cpu', dtype=torch.uint8)
        self.assertEqual(y_mps, y_cpu)

    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    # See https://github.com/pytorch/pytorch/issues/84995
    def test_div_bugs(self):
        for (dtype, mode) in itertools.product(integral_types(), ['trunc', 'floor']):
            if dtype != torch.int64:
                x = torch.tensor(list(range(1, 11)), device='mps', dtype=dtype)
                y = torch.div(x, 101, rounding_mode=mode)
                self.assertEqual(y.sum(), 0)

    # See https://github.com/pytorch/pytorch/issues/82663
    def test_bool_expand(self):
        x = torch.tensor([[1], [0]], dtype=torch.bool, device='mps')
        y = torch.tensor([0, 1], dtype=torch.bool, device='mps')
        self.assertFalse(torch.equal(x.expand(2, 2), y.expand(2, 2)))

    # Empty unary op should return tensor of the same size
    def test_empty_neg(self):
        x = torch.tensor([[]], device='mps')
        y = -x
        self.assertEqual(x, y)

    def _test_unique_scalar_empty(self, dtype, device, f):
        # test scalar
        x = torch.tensor(0, dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        expected_inverse = torch.tensor(0, device=device)
        expected_counts = torch.tensor([1], device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

        # test zero sized tensor
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

    def _test_unique_with_expects(self, device, dtype, f, x, expected_unique, expected_inverse, expected_counts, additional_shape):
        def ensure_tuple(x):
            if isinstance(x, torch.Tensor):
                return (x,)
            return x

        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                # test with expected
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                self.assertEqual(expected_unique, ret[0])
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])

                # tests per-element unique on a higher rank tensor.
                y = x.view(additional_shape)
                y_unique, y_inverse, y_counts = f(y, return_inverse=True, return_counts=True)
                self.assertEqual(expected_unique, y_unique)
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                self.assertEqual(expected_counts, y_counts)

    def test_unique_all_dtypes(self, device="mps"):
        def helper(dtype):
            def ensure_tuple(x):
                if isinstance(x, torch.Tensor):
                    return (x,)
                return x

            if dtype is torch.bool:
                x = torch.tensor([True, False, False, False, True, False, True, False], dtype=torch.bool, device=device)
                expected_unique = torch.tensor([False, True], dtype=torch.bool, device=device)
                expected_inverse = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.long, device=device)
                expected_counts = torch.tensor([5, 3], dtype=torch.long, device=device)
            else:
                x = torch.tensor([1, 2, 3, 2, 8, 5, 2, 3], dtype=dtype, device=device)
                expected_unique = torch.tensor([1, 2, 3, 5, 8], dtype=dtype, device=device)
                expected_inverse = torch.tensor([0, 1, 2, 1, 4, 3, 1, 2], device=device)
                expected_counts = torch.tensor([1, 3, 2, 1, 1], device=device)

            # test sorted unique
            fs = (
                lambda x, **kwargs: torch.unique(x, sorted=True, **kwargs),
                lambda x, **kwargs: x.unique(sorted=True, **kwargs),
            )
            x_sliced = torch.empty(x.size(0) * 2, dtype=dtype, device=device)[::2].copy_(x)
            xs = (x, x_sliced)
            for f, x in product(fs, xs):
                self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (2, 2, 2))
                self._test_unique_scalar_empty(dtype, device, f)

            # test unsorted unique
            fs = (
                lambda x, **kwargs: torch.unique(x, sorted=False, **kwargs),
                lambda x, **kwargs: x.unique(sorted=False, **kwargs)
            )
            for f, x in product(fs, xs):
                self._test_unique_scalar_empty(dtype, device, f)
                for return_inverse, return_counts in product((True, False), repeat=2):
                    ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                    self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                    x_list = x.tolist()
                    x_unique_list = ret[0].tolist()
                    self.assertEqual(expected_unique.tolist(), sorted(x_unique_list))
                    if return_inverse:
                        x_inverse_list = ret[1].tolist()
                        for i, j in enumerate(x_inverse_list):
                            self.assertEqual(x_list[i], x_unique_list[j])
                    if return_counts:
                        count_index = 1 + int(return_inverse)
                        x_counts_list = ret[count_index].tolist()
                        for i, j in zip(x_unique_list, x_counts_list):
                            count = 0
                            for k in x_list:
                                if k == i:
                                    count += 1
                            self.assertEqual(j, count)
        [helper(dtype) for dtype in [torch.float32, torch.int64, torch.int32, torch.int16, torch.uint8]]

    def test_unique(self):
        def helper(x, return_inverse, return_counts):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)
            result_cpu = torch.unique(cpu_x, return_inverse=return_inverse, return_counts=return_counts)

            self.assertEqual(result, result_cpu)
        helper(torch.tensor([1, 2, 4, 2, 1]), False, False)
        helper(torch.randint(3, (10, )), False, False)
        helper(torch.randint(3, (10, )), True, False)
        helper(torch.randint(3, (10, )), False, True)
        helper(torch.randint(3, (10, )), True, True)
        helper(torch.randint(3, (1, )), True, True)
        helper(torch.randint(3, (0, )), True, True)
        # Regression test for https://github.com/pytorch/pytorch/issues/104879
        x = torch.arange(2, device="mps")
        self.assertEqual(x.reshape(1, 1, 2).unique(), x)

    def test_unique_consecutive(self):
        def helper(x, dim, return_inverse, return_counts):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.unique_consecutive(x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)
            result_cpu = torch.unique_consecutive(cpu_x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)

            self.assertEqual(result, result_cpu)
        helper(torch.tensor([1, 2, 4, 2, 1]), 0, False, False)
        helper(torch.randint(3, (10, )), 0, False, False)
        helper(torch.randint(3, (10, )), 0, True, False)
        helper(torch.randint(3, (10, )), 0, False, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (1, )), 0, True, True)
        helper(torch.randint(3, (0, )), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, True, True)
        helper(torch.randint(2, (20, 2)), 0, True, True)
        helper(torch.randint(2, (1, 2)), 0, True, True)
        helper(torch.randint(2, (0, 2)), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, True, True)
        helper(torch.randint(2, (2, 20)), 1, True, True)
        helper(torch.randint(2, (2, 1)), 1, True, True)
        helper(torch.randint(2, (2, 0)), 1, True, True)

    # See https://github.com/pytorch/pytorch/issues/85675
    def test_cat_non_contiguous(self):
        def rotate_subset(data, dim):
            x1 = data[:, :, :2, :]
            x2 = data[:, :, 2:, :]
            self.assertFalse(x1.is_contiguous())
            self.assertFalse(x2.is_contiguous())
            return torch.concat((x1, x2), dim=dim)
        for dtype in MPS_DTYPES:
            if dtype == torch.bool:
                continue
            data = torch.arange(48, dtype=dtype).reshape(1, 2, 4, 6)
            data = data.to(memory_format=torch.channels_last)
            mps_data = data.to("mps")
            self.assertEqual(data, mps_data)
            for dim in range(data.dim()):
                cpu_result = rotate_subset(data, dim)
                mps_result = rotate_subset(mps_data, dim)
                self.assertEqual(cpu_result, mps_result.to("cpu"))
                # TODO: enable memory format test
                # self.assertEqual(cpu_result.is_contiguous(), mps_result.is_contiguous())

    # See https://github.com/pytorch/pytorch/issues/85967
    def test_from_numpy_non_contiguous(self):
        a = np.arange(9).reshape(3, 3)[:, :2]
        t_cpu = torch.tensor(a, device="cpu")
        t_mps = torch.tensor(a, device="mps")
        self.assertEqual(t_cpu, t_mps.to("cpu"))

    # See https://github.com/pytorch/pytorch/issues/86954
    def test_copy_non_contiguous(self):
        x = torch.arange(27).reshape(3, 3, 3).permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        y = x.to('mps')
        self.assertFalse(y.is_contiguous())
        self.assertEqual(x, y.to('cpu'))

        x = torch.arange(4**3).reshape(4, 4, 4).permute((2, 0, 1))[1:, ::2]
        y = x.to('mps')
        self.assertEqual(x, y.to('cpu'))

        x = torch.full((4, 4, 4, 4), 13, device="cpu")
        y = torch.full((4, 4, 4, 4), 13, device="mps")
        z = torch.arange(4**4).reshape(4, 4, 4, 4).permute(3, 2, 0, 1)[1::, ::2]
        x.permute(3, 2, 1, 0)[1::, ::2] = z
        # As y is on MPS and z on CPU, this dispatches to a copy operator
        y.permute(3, 2, 1, 0)[1::, ::2] = z
        self.assertEqual(x, y.to('cpu'))

    # See https://github.com/pytorch/pytorch/issues/95417
    def test_copy_storage_offset(self):
        x_cpu = torch.zeros(5, device="cpu", dtype=torch.float32)
        x_mps = torch.zeros(5, device="mps", dtype=torch.float32)
        update_cpu = torch.tensor([1, 1], device="cpu", dtype=torch.int64)
        update_mps = torch.tensor([1, 1], device="mps", dtype=torch.int64)
        x_cpu[2:4] = update_cpu
        x_mps[2:4] = update_mps  # implicit type casting and copy
        self.assertEqual(x_cpu, x_mps)

        x_cpu[2:4] = update_mps  # implicit device moving and copy
        self.assertEqual(x_cpu, x_mps)

    def test_copy_broadcasting(self):
        def helper(src_shape, dst_shape, src_dtype, dst_dtype):
            cpu_src = torch.randint(0, 127, src_shape).to(src_dtype)
            cpu_dst = torch.randint(0, 127, dst_shape).to(dst_dtype)
            cpu_result = cpu_dst.copy_(cpu_src)
            mps_src = cpu_src.to("mps")
            mps_dst = cpu_dst.to("mps")
            mps_result = mps_dst.copy_(mps_src)
            self.assertEqual(cpu_result, mps_result)

        test_dtypes = [torch.float32, torch.int32, torch.int16, torch.int8]

        for (src_dtype, dst_dtype) in itertools.product(test_dtypes, test_dtypes):
            helper((2, 1), (2, 3), src_dtype, dst_dtype)
            helper((2, 1), (2, 2), src_dtype, dst_dtype)
            helper((3, 1, 4, 1), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((3,), (2, 3), src_dtype, dst_dtype)
            helper((2,), (2, 2), src_dtype, dst_dtype)
            helper((4, 1, 5), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((4, 1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((3, 1, 0), (3, 5, 0), src_dtype, dst_dtype)
            helper((0, 1, 0), (0, 5, 0), src_dtype, dst_dtype)
        # Regression test for https://github.com/pytorch/pytorch/issues/107867
        self.assertEqual(torch.tensor([[1]], device='mps').item(), 1.0)

    # See https://github.com/pytorch/pytorch/pull/84742
    # and https://github.com/pytorch/pytorch/pull/78319
    def test_binops_dtype_precedence(self):
        # Test dtype precedence (casting order) in binary operations by comparing to CPU result
        # Example values for all dtypes supported on the MPS backend
        sample_vals = {
            torch.bool: [False, True],
            torch.int16: [-15, 0, 1, 10],
            torch.int32: [-376, 0, 1, 13],
            torch.int64: [-8, 0, 1, 77],
            torch.float16: [-234.5, 0.0, 1.0, 2.0],
            torch.float32: [-1.0, 0.0, 0.1, 111.99],
        }
        # Test all combinations of dtypes, operations, dimensionality
        for dtype1, dtype2, binop in itertools.product(
                sample_vals.keys(), sample_vals.keys(), ['add', 'sub', 'mul', 'div']):
            # bool minus bool is generally unsupported, so skip
            if binop == 'sub' and (dtype1 == torch.bool or dtype2 == torch.bool):
                continue
            full_shape = (10,)
            for val1, val2 in itertools.product(sample_vals[dtype1], sample_vals[dtype2]):
                # print(f'{dtype1},{dtype2}: ({val1}).{binop}({val2})')
                # print(getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                #            (torch.tensor(val2, dtype=dtype2, device='mps')))
                # print(getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                #            (torch.tensor(val2, dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor([val1], dtype=dtype1, device='mps'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='mps')),
                    getattr(torch.tensor([val1], dtype=dtype1, device='cpu'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor([val1], dtype=dtype1, device='mps'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor([val1], dtype=dtype1, device='cpu'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='cpu')))
                # Test tensors created with torch.full
                x1 = torch.full(full_shape, val1, dtype=dtype1, device='mps')
                y1 = torch.tensor(val2, dtype=dtype2, device='mps')
                x2 = torch.full(full_shape, val1, dtype=dtype1, device='cpu')
                y2 = torch.tensor(val2, dtype=dtype2, device='cpu')
                self.assertEqual(getattr(x1, binop)(y1), getattr(x2, binop)(y2))
                x3 = torch.tensor(val1, dtype=dtype1, device='mps')
                y3 = torch.full(full_shape, val2, dtype=dtype2, device='mps')
                x4 = torch.tensor(val1, dtype=dtype1, device='cpu')
                y4 = torch.full(full_shape, val2, dtype=dtype2, device='cpu')
                self.assertEqual(getattr(x3, binop)(y3), getattr(x4, binop)(y4))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.full(full_shape, val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.full(full_shape, val2, dtype=dtype2, device='cpu')))

    def test_nansum(self):
        def helper(dtype, noncontiguous, dim):
            zero_cpu = torch.zeros((), dtype=dtype)

            # Randomly scale the values
            scale = random.randint(10, 100)
            x_cpu: torch.Tensor = make_tensor(
                (5, 5), dtype=dtype, device='cpu',
                low=-scale, high=scale, noncontiguous=noncontiguous)

            if dtype.is_floating_point:
                nan_mask_cpu = x_cpu < (0.2 * scale)
                x_no_nan_cpu = torch.where(nan_mask_cpu, zero_cpu, x_cpu)
                x_cpu[nan_mask_cpu] = np.nan
            else:
                x_no_nan_cpu = x_cpu

            x_mps = x_cpu.to('mps')
            actual_out_mps = torch.empty(0, dtype=dtype, device='mps')
            expect_out_cpu = torch.empty(0, dtype=dtype)
            dim_kwargs = {"dim": dim} if dim is not None else {}
            expect = torch.sum(x_no_nan_cpu, **dim_kwargs)

            actual_cpu = torch.nansum(x_cpu, **dim_kwargs)
            # Sanity check on CPU
            self.assertEqual(expect, actual_cpu)

            # Test MPS
            actual_mps = torch.nansum(x_mps, **dim_kwargs)
            # Test out= variant
            torch.nansum(x_mps, out=actual_out_mps, **dim_kwargs)
            torch.nansum(x_cpu, out=expect_out_cpu, **dim_kwargs)
            self.assertEqual(expect, actual_mps)
            self.assertEqual(expect_out_cpu, actual_out_mps)

        args = itertools.product(
            (torch.float16, torch.float32, torch.int32, torch.int64),   # dtype
            (True, False),                                              # noncontiguous
            (0, 1, None),                                               # dim
        )

        for dtype, noncontiguous, dim in args:
            with self.subTest(dtype=dtype, noncontiguous=noncontiguous, dim=dim):
                helper(dtype, noncontiguous, dim)

    def test_cumsum_all_dtypes(self):
        def helper(dtype):
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            a = t.cumsum(0, dtype=dtype)
            a_cpu = t_cpu.cumsum(0, dtype=dtype)

            self.assertEqual(a.cpu(), a_cpu)
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        try:
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            self.assertEqual(e_string, "MPS does not support cumsum_out_mps op with int64 input." +
                             " Support has been added in macOS 13.3")

    def test_cumsum_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            cpu_x = None
            if(dtype == torch.float32):
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = cpu_x.cumsum(-1)
            y = x.cumsum(-1)

            self.assertEqual(y, cpu_y)

        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_cumprod_all_dtypes(self):
        def helper(dtype):
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            a = t.cumprod(0, dtype=dtype)
            a_cpu = t_cpu.cumprod(0, dtype=dtype)

            self.assertEqual(a.cpu(), a_cpu)
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        try:
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            self.assertEqual(e_string, "MPS does not support cumprod_out_mps op with int64 input."
                             + " Support has been added in macOS 13.3")

    def test_cumprod_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            cpu_x = None
            if(dtype == torch.float32):
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = cpu_x.cumprod(-1)
            y = x.cumprod(-1)

            self.assertEqual(y, cpu_y)

        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_median_int16(self):
        def helper(shape, dtype):
            cpu_x = torch.randint(-9999, 9999, shape, device='cpu', dtype=dtype)
            x = cpu_x.detach().clone().to('mps')

            median_result = torch.median(x)
            median_result_cpu = torch.median(cpu_x)
            self.assertEqual(median_result, median_result_cpu)

        helper((2, 8, 4, 5), torch.int16)

    def test_activation_checkpoint_does_not_error(self):
        from torch.utils.checkpoint import checkpoint

        for use_reentrant in (True, False):
            a = torch.tensor(1., device="mps", requires_grad=True)

            def fn(x):
                return x.sin().cos().exp()

            out = checkpoint(fn, a, use_reentrant=use_reentrant)
            out.backward()

class TestLogical(TestCaseMPS):
    def _wrap_tensor(self, x, device="cpu", dtype=None, requires_grad=False):
        return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)

    def test_logical_not(self):
        def helper(x):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.logical_not(x)
            result_cpu = torch.logical_not(cpu_x)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]))
        helper(self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True))
        helper(self._wrap_tensor([True, True, False, False]))
        helper(self._wrap_tensor(1))
        helper(self._wrap_tensor(0))
        helper(self._wrap_tensor(True))
        helper(self._wrap_tensor(False))

    def test_logical_and(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_and(x, other)
            result_cpu = torch.logical_and(cpu_x, cpu_other)
            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_logical_or(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_or(x, other)
            result_cpu = torch.logical_or(cpu_x, cpu_other)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_logical_xor(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_xor(x, other)
            result_cpu = torch.logical_xor(cpu_x, cpu_other)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_min_max(self):
        def helper(dtype):
            for _ in range(10):
                if dtype == torch.float32 or dtype == torch.float16:
                    x = torch.randn((30, 15), device='mps', dtype=dtype)
                else:
                    x = torch.randint(0, 100, (30, 15), device="mps", dtype=dtype)
                x_cpu = x.to("cpu")

                y = x.max()
                y_cpu = x_cpu.max()
                self.assertEqual(y, y_cpu)

                z = x.min()
                z_cpu = x_cpu.min()
                self.assertEqual(z, z_cpu)

        [helper(dtype) for dtype in [torch.float32, torch.float16, torch.int32, torch.int16, torch.uint8, torch.int8, torch.bool]]

class TestSmoothL1Loss(TestCaseMPS):

    def _smooth_l1_loss_helper(self, reduction="mean", requires_grad=False):
        # CPU
        input_cpu = torch.randn(4, 7, requires_grad=requires_grad)
        target_cpu = torch.randn(4, 7)

        # MPS
        input_mps = input_cpu.detach().clone().to('mps').requires_grad_()
        target_mps = target_cpu.detach().clone().to('mps')

        smooth_l1_loss_cpu = F.smooth_l1_loss(input_cpu, target_cpu, beta=1.0, reduction=reduction)
        smooth_l1_loss_mps = F.smooth_l1_loss(input_mps, target_mps, beta=1.0, reduction=reduction)

        self.assertEqual(smooth_l1_loss_cpu, smooth_l1_loss_mps)

        if requires_grad:
            smooth_l1_loss_cpu.backward()
            smooth_l1_loss_mps.backward()
            self.assertEqual(input_cpu.grad, input_mps.grad.to("cpu"))

        return smooth_l1_loss_cpu, smooth_l1_loss_mps

    def test_smooth_l1_loss_reduction_none(self):
        self._smooth_l1_loss_helper(reduction="none")

    def test_smooth_l1_loss_reduction_mean(self):
        self._smooth_l1_loss_helper(reduction="mean")

    def test_smooth_l1_loss_reduction_sum(self):
        self._smooth_l1_loss_helper(reduction="sum")

    def test_smooth_l1_loss_reduction_mean_backward(self):
        self._smooth_l1_loss_helper(reduction="mean", requires_grad=True)

    def test_smooth_l1_loss_reduction_mean_sum_backward(self):
        self._smooth_l1_loss_helper(reduction="sum", requires_grad=True)


class TestNLLLoss(TestCaseMPS):
    def test_nll_loss_mismatched_batch(self, device='mps'):
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self):

        def test_nll_loss_out_of_bounds_ignore_index_helper(device):
            output = []
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t1 = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
            t2 = torch.tensor([0, 1, 1, 0, -100, 2], dtype=torch.int64, device=device)
            for reduction in ['mean', 'none']:
                # out of bound ignore_index
                output.append(F.nll_loss(x, t1, ignore_index=255, reduction=reduction))
                # default ignore_index
                output.append(F.nll_loss(x, t2, reduction=reduction))
            return output

        output_cpu = test_nll_loss_out_of_bounds_ignore_index_helper(device='cpu')
        output_mps = test_nll_loss_out_of_bounds_ignore_index_helper(device='mps')

        for cpu, mps in zip(output_cpu, output_mps):
            self.assertEqual(cpu, mps)

    def test_nll_loss_invalid_target_dim(self):

        def _test_nll_loss_invalid_target_dim(device):
            output = []
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t = torch.zeros((6, 2), dtype=torch.int64, device=device)
            with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
                F.nll_loss(x, t)

        _test_nll_loss_invalid_target_dim(device='cpu')
        _test_nll_loss_invalid_target_dim(device='mps')

    def test_nll_loss_invalid_weights(self):

        def _test_nll_loss_invalid_weights(device):
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t = torch.tensor([0, 1, 2, 1, 1, 2], dtype=torch.int64, device=device)
            invalid_weights = [
                torch.zeros(4, device=device),
                torch.zeros((1, 3), device=device),
            ]
            msg = "weight tensor should be defined either for all 3 classes or no classes"
            for weight in invalid_weights:
                with self.assertRaisesRegex(RuntimeError, msg):
                    F.nll_loss(x, t, weight=weight)

        _test_nll_loss_invalid_weights(device='cpu')
        _test_nll_loss_invalid_weights(device='mps')

    def _nll_loss_helper(self, input_size, reduction, expected):

        # CPU
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device='cpu')
        weights = torch.randn(num_channels)

        # MPS
        input_mps = input.detach().clone().to('mps').requires_grad_()
        target_mps = target.detach().clone().to('mps')
        weights_mps = weights.to("mps")

        output_cpu = F.nll_loss(input, target, weight=weights, reduction=reduction)
        output_mps = F.nll_loss(input_mps, target_mps, weight=weights_mps, reduction=reduction)
        self.assertEqual(output_cpu, output_mps.to('cpu'))

        output_cpu.sum().backward()
        output_mps.sum().backward()
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))

    def _nll_loss_1d_helper(self, input_size, reduction):

        # CPU
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        num_channels = input_size[0]
        target = torch.randint(num_channels, [], device='cpu')

        # MPS
        input_mps = input.detach().clone().to('mps').requires_grad_()
        target_mps = target.detach().clone().to('mps')

        output_cpu = F.nll_loss(input, target, reduction=reduction)
        output_mps = F.nll_loss(input_mps, target_mps, reduction=reduction)
        self.assertEqual(output_cpu, output_mps.to('cpu'))

        output_cpu.sum().backward()
        output_mps.sum().backward()
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))

    def test_as_strided(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        values_1 = [[1.0, 1.0], [1.0, 1.0]]
        cpu_x = torch.tensor(values, device='cpu')
        ones1 = torch.tensor(values_1, device='mps')
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        strided_cpu = torch.as_strided(cpu_x, (2, 2), (1, 2))
        strided_mps = torch.as_strided(x, (2, 2), (1, 2))
        self.assertEqual(strided_mps, strided_cpu)
        strided_cpu_out = strided_cpu + ones1.to('cpu')
        strided_mps_out = strided_mps + ones1
        self.assertEqual(strided_cpu_out, strided_mps_out)

        # test with storage offsets
        cpu_x = torch.rand(3, 3, device='cpu')
        mps_x = cpu_x.to('mps')
        strided_cpu1 = torch.as_strided(cpu_x, (2, 2), (1, 2), 0)
        strided_mps1 = torch.as_strided(mps_x, (2, 2), (1, 2), 0)
        strided_cpu2 = torch.as_strided(cpu_x, (2, 2), (1, 2), 1)
        strided_mps2 = torch.as_strided(mps_x, (2, 2), (1, 2), 1)
        strided_cpu_out = strided_cpu1 - strided_cpu2
        strided_mps_out = strided_mps1 - strided_mps2
        self.assertEqual(strided_cpu_out, strided_mps_out)

    def test_unfold(self):
        x = torch.arange(1., 8)
        x_mps = torch.arange(1., 8, device="mps")

        y = x.unfold(0, 2, 1)
        y_mps = x_mps.unfold(0, 2, 1)

        self.assertEqual(y, y_mps)

    def test_unfold_all_devices_and_dtypes(self):
        supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
        for dt in supported_dtypes:
            x = torch.empty((0, 1, 3, 0), dtype=dt, device="mps")
            self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    def test_unfold_scalars(self):
        x = torch.tensor(0.5, device="mps")
        # unfold on a 0-dimensional tensor should always return a 1-d dimensional
        # tensor of shape [size] (i.e., the second parameter to unfold)

        self.assertEqual(torch.empty(0, device="mps"), x.unfold(0, 0, 1))
        self.assertEqual(torch.empty(0, device="mps"), x.unfold(0, 0, 2))
        self.assertEqual(torch.tensor([0.5], device="mps"), x.unfold(0, 1, 1))

    def test_bincount_simple(self):
        input = torch.randint(0, 8, (5,), dtype=torch.int32, device="mps")
        input_cpu = input.to("cpu")
        weights = torch.linspace(0, 1, steps=5, device="mps", dtype=torch.float32)
        weights_cpu = weights.to("cpu")

        x = torch.bincount(input)
        x_cpu = torch.bincount(input_cpu)
        self.assertEqual(x, x_cpu)

        y = input.bincount(weights)
        y_cpu = input_cpu.bincount(weights_cpu)
        self.assertEqual(y, y_cpu)

    def test_bincount_reduction(self):
        device = "mps"
        # negative input throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([1, -1], device=device, dtype=torch.int32))
        # n-d input, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([[1, 2], [3, 4]], device=device))
        # minlength < 0 throws
        with self.assertRaisesRegex(RuntimeError, 'minlength should be >= 0'):
            torch.bincount(torch.tensor([1, 3], device=device),
                           torch.tensor([.2, .2], device=device),
                           minlength=-1)
        # n-d weights, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d'):
            torch.bincount(torch.tensor([1, 0], device=device, dtype=torch.int32),
                           torch.tensor([[1., 0.3], [1., 0.3]], device=device, dtype=torch.float))
        # input and weights dim mismatch
        with self.assertRaisesRegex(RuntimeError, 'same length'):
            torch.bincount(torch.tensor([1, 0], device=device, dtype=torch.int32),
                           torch.tensor([1., 0.3, 0.5], device=device, dtype=torch.float))
        # 1-d input with no elements and default minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long)),
                         torch.zeros(0, dtype=torch.long, device=device))
        # 1-d input with no elements and specified minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long), minlength=10),
                         torch.zeros(10, dtype=torch.long, device=device))

        # test tensor method without weights
        long_counts = torch.tensor(
            [0, 3, 2, 1, 3], dtype=torch.uint8, device=device).bincount()
        self.assertEqual(
            torch.tensor([1, 1, 1, 2], dtype=torch.int64, device=device),
            long_counts)
        # test avoiding overflow for uint8 (#76979)
        count_uint8 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.uint8, device=device).bincount()
        count_int16 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.int16, device=device).bincount()
        self.assertEqual(count_uint8, count_int16)
        # test minlength functionality
        int_counts = torch.bincount(
            torch.tensor([1, 1, 1, 1], device=device, dtype=torch.int32), minlength=5)
        self.assertEqual(
            torch.tensor([0, 4, 0, 0, 0], dtype=torch.int64, device=device),
            int_counts)
        # test weights
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device, dtype=torch.int32),
            torch.tensor([.1, .2, .3, .4, .5], device=device))
        self.assertEqual(
            torch.tensor([0.1, 0.9, 0, 0, 0.5], device=device), byte_counts)
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device, dtype=torch.int32),
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device=device))
        self.assertEqual(
            torch.tensor([1, 9, 0, 0, 5], device=device, dtype=torch.int32), byte_counts)
        # test non-contiguous inputs and weights
        inputs = torch.tensor([[0, 0], [3, 1], [2, 1], [1, 1], [3, 4]], device=device, dtype=torch.int32)
        weights = torch.tensor([[.1, 1], [.2, 2], [.3, 3], [.4, 4], [.5, 5]], device=device)
        for i in [0, 1]:
            assert not inputs[:, i].is_contiguous(), "Inputs are supposed to be non-contiguous"
            assert not weights[:, i].is_contiguous(), "Weights are supposed to be non-contiguous"
        # inputs are non-contiguous but weights are contiguous
        self.assertEqual(inputs[:, 0].bincount(), torch.tensor([1, 1, 1, 2]))
        # inputs and weights are non-contiguous
        self.assertEqual(
            inputs[:, 1].bincount(weights[:, 1]),
            torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))
        # weights are non-contiguous but inputs are contiguous
        self.assertEqual(inputs[:, 1].contiguous().bincount(weights[:, 1]),
                         torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))

        # test bincount on non-contiguous slices
        all0s = torch.zeros((32, 2), dtype=torch.int32, device=device)
        self.assertEqual(all0s[:, 0].bincount(), torch.tensor([32]))

        all1s = torch.ones((32, 2), dtype=torch.int32, device=device)
        self.assertEqual(all1s[:, 0].bincount(), torch.tensor([0, 32]))

        # test large number of bins - global memory use
        big_exp = torch.zeros(100, device=device)
        big_exp[-1] = 50.0
        big_w = torch.tensor([.5] * 100, device=device)
        big_out = torch.tensor([99] * 100, device=device, dtype=torch.int32).bincount(big_w)
        self.assertEqual(big_exp, big_out)
        # test large input size
        big_exp = torch.zeros(2, device=device, dtype=torch.int64)
        big_exp[1] = 10
        big_out = torch.ones(10, dtype=torch.int8, device=device).bincount()
        self.assertEqual(big_exp, big_out)

    def test_bincount(self):
        device = "mps"
        input_size = (5000,)
        w = torch.randn(input_size, dtype=torch.float, device=device)
        w_cpu = w.cpu()

        t = torch.randint(50, input_size, dtype=torch.int8, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.randint(500, input_size, dtype=torch.int32, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.randint(2000, input_size, dtype=torch.int32, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device=device)
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_sum_backward(self):
        def helper(n, c):
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            all_sum.backward()
            all_sum_cpu.backward()
            self.assertEqual(all_sum, all_sum_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper(3, 3)

    def test_nll_loss_1d(self, device='cpu'):
        self._nll_loss_1d_helper([10], "none")
        self._nll_loss_1d_helper([10], "mean")
        self._nll_loss_1d_helper([10], "sum")

    def test_nll_loss_empty_tensor_reduction_none(self, device='cpu'):
        self._nll_loss_helper([1, 3], "none", torch.empty([0], device=device))
        self._nll_loss_helper([3, 5, 7], "none", torch.empty([5, 7], device=device))
        self._nll_loss_helper([2, 3, 1, 7], "none", torch.empty([2, 1, 7], device=device))
        self._nll_loss_helper([2, 3, 5, 1], "none", torch.empty([2, 5, 1], device=device))
        self._nll_loss_helper([2, 3, 5, 7, 1], "none", torch.empty([2, 5, 7, 1], device=device))

    def test_nll_loss_empty_tensor_reduction_mean(self, device='cpu'):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([1, 3], "mean", nan)
        self._nll_loss_helper([1, 3, 5, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 1, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 1], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 7, 1], "mean", nan)

    def test_nll_loss_empty_tensor_reduction_sum(self, device='cpu'):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([1, 3], "sum", zero)
        self._nll_loss_helper([1, 3, 5, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 1, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 1], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 7, 1], "sum", zero)

    def test_nll_loss_byte_target_matches_long(self, device='cpu'):
        N, C = 10, 4
        input = torch.randn(N, C, device=device, requires_grad=True)
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        def compute_result_and_gradient(reduction, target_dtype):
            result, grad = {}, {}
            for dev in ['cpu', 'mps']:
                input_dev = input.to(dev)
                input_ = input_dev.detach()
                input_.requires_grad_()

                target_dev = target.to(dev)

                prob = F.log_softmax(input_, dim=-1)
                loss = nn.NLLLoss(reduction=reduction)
                result[dev] = loss(prob, target_dev.to(target_dtype))
                result[dev].sum().backward()
                grad[dev] = input_.grad

            return result, grad

        for reduction in ["none", "mean", "sum"]:
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)

            self.assertEqual(result_long['mps'].to('cpu'), result_long['cpu'])
            self.assertEqual(grad_long['mps'].to('cpu'), grad_long['cpu'])

    # L1 loss
    def test_l1_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.L1Loss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # Mean Squared Error
    def test_mse_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.MSELoss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # Binary Cross Enropy
    def test_bce_loss_simple(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.BCELoss(reduction=reduction)

            # input and target must be within [0..1]
            input_t = np.random.random_sample(size=shape).astype(np.float32)
            target_t = np.random.random_sample(size=shape).astype(np.float32)
            inputCPU = torch.tensor(input_t, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.tensor(target_t, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 0.6 just to have the grad_output != 1
                outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')
        helper([1, 1, 32, 32], 'mean')

    def test_bce_loss_always_nonnegative(self):
        target = torch.ones(5, device='mps')
        input = torch.ones(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        target = torch.zeros(5, device='mps')
        input = torch.zeros(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    def test_bce_loss_size_mismatch(self):
        bceloss = nn.BCELoss()
        a = torch.rand(25, device='mps')
        b = torch.rand(25, 1, device='mps')
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        x_size = 1024
        y_size = 256
        target = torch.rand(x_size, y_size, device='mps')

        for reduction in ['none', 'mean', 'sum']:
            output_sig = torch.rand(x_size, y_size, device='mps') - 0.5
            output_logits = output_sig.clone().detach()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = torch.rand(y_size, device='mps')

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = torch.rand(x_size, y_size, device='mps')
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.ones(64, 4, device='mps')

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        pos_weight = torch.ones(3, 1, device='mps')
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self):
        output = torch.tensor([0., -120.], device='mps')
        target = torch.tensor([0., 1.], device='mps')
        pos_weight = torch.tensor([1., 1.], device='mps')

        out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(torch.isfinite(out1).all().item())

        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(torch.isfinite(out2).all().item())

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

    def test_log_softmax(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        cpu_log_softmax = F.log_softmax(cpu_x, dim=0)
        mps_log_softmax = F.log_softmax(mps_x, dim=0)
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        cpu_grad = torch.ones_like(cpu_log_softmax)
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        cpu_log_softmax.backward(gradient=cpu_grad)
        mps_log_softmax.backward(gradient=mps_grad)

        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    def test_log_softmax_large_numbers(self):
        values = [
            [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
            [-10.0, -100.0, -1000.0, -10000.0, -100000.0, -1000000.0]
        ]
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        cpu_log_softmax = F.log_softmax(cpu_x, dim=-1)
        mps_log_softmax = F.log_softmax(mps_x, dim=-1)
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        cpu_grad = torch.ones_like(cpu_log_softmax)
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        cpu_log_softmax.backward(gradient=cpu_grad)
        mps_log_softmax.backward(gradient=mps_grad)

        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    def test_eq(self):
        values1 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        values2 = [[[1.0, 2.0, 15.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 11.0, 12.0]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    def test_signed_vs_unsigned_comparison(self):
        cpu_x = torch.tensor((-1, 2, 3), device='cpu', dtype=torch.uint8)
        mps_x = torch.tensor((-1, 2, 3), device='mps', dtype=torch.uint8)
        # in the comparison of signed vs. unsigned we should always cast to unsigned
        self.assertEqual(cpu_x == -1, mps_x == -1)
        self.assertEqual(cpu_x > -1, mps_x > -1)
        self.assertEqual(cpu_x < -1, mps_x < -1)

    def test_eq_int64(self):
        values1 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        values2 = [[[1, 2, 15], [4, 5, 6]], [[7, 8, 9], [0, 11, 12]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_ne(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, mps_y)
            result_cpu = torch.ne(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ne_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, 0.0)
            result_cpu = torch.ne(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, mps_y)
            result_cpu = torch.lt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, 0.0)
            result_cpu = torch.lt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.le(mps_x, mps_y)
            result_cpu = torch.le(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.le(mps_x, 0.0)
            result_cpu = torch.le(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, mps_y)
            result_cpu = torch.ge(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, 0.0)
            result_cpu = torch.ge(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, mps_y)
            result_cpu = torch.gt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, 0.0)
            result_cpu = torch.gt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_argmax(self):
        # https://github.com/pytorch/pytorch/issues/98191
        cpu_tensor = torch.tensor([[0, 1], [2, 1], [1, 0]])
        res_cpu = torch.argmax(cpu_tensor, dim=1)

        mps_tensor = cpu_tensor.to(torch.device('mps'))
        res_mps = torch.argmax(mps_tensor, dim=1)
        self.assertEqual(res_cpu, res_mps)

        # https://github.com/pytorch/pytorch/issues/92311
        mps_tensor = torch.randn(10, 2, device='mps', dtype=torch.float32)
        cpu_tensor = mps_tensor.detach().clone().cpu()

        res_mps = torch.argmax(mps_tensor, dim=1)
        res_cpu = torch.argmax(cpu_tensor, dim=1)
        self.assertEqual(res_cpu, res_mps)

    # Test forward argmin argmax
    def test_argmin_argmax(self):
        def helper(n, c, h, w, reduction_type, dtype=torch.float32):
            if reduction_type == "max":
                arg_reduction_fn = torch.argmax
            else:
                arg_reduction_fn = torch.argmin

            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = arg_reduction_fn(x)
            ref_y = arg_reduction_fn(cpu_x)
            self.assertEqual(y, ref_y)

            y_0 = arg_reduction_fn(x, dim=0)
            refy_0 = arg_reduction_fn(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)

            y_0dim = arg_reduction_fn(x, dim=0, keepdim=True)
            refy_0dim = arg_reduction_fn(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)

            y_1 = arg_reduction_fn(x, dim=1)
            refy_1 = arg_reduction_fn(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)

            y_1dim = arg_reduction_fn(x, dim=1, keepdim=True)
            refy_1dim = arg_reduction_fn(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)

            y_2 = arg_reduction_fn(x, dim=2)
            refy_2 = arg_reduction_fn(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)

            y_2dim = arg_reduction_fn(x, dim=2, keepdim=True)
            refy_2dim = arg_reduction_fn(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)

            y_3 = arg_reduction_fn(x, dim=3)
            refy_3 = arg_reduction_fn(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)

            y_3dim = arg_reduction_fn(x, dim=3, keepdim=True)
            refy_3dim = arg_reduction_fn(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)

        helper(2, 8, 4, 4, "max", torch.float32)
        helper(2, 8, 4, 4, "max", torch.int32)
        helper(2, 8, 4, 4, "max", torch.float16)
        helper(2, 8, 4, 4, "max", torch.int64)
        helper(2, 8, 4, 4, "min", torch.float32)
        helper(2, 8, 4, 4, "min", torch.int32)
        helper(2, 8, 4, 4, "min", torch.float16)
        helper(2, 8, 4, 4, "min", torch.int64)

    @unittest.skipIf(product_version < 13.3, "Long data type supported from macOS 13.3 and above")
    def test_reduction_sum_max_long_val(self):
        x_mps = torch.tensor([sys.maxsize, sys.maxsize - 10, sys.maxsize - 5, sys.maxsize - 18], device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        res_mps = torch.sum(x_mps)
        res_cpu = torch.sum(x_cpu)
        self.assertEqual(res_mps, res_cpu)

    # Test forward max
    # Note - don't test grad now
    def test_max_el(self):
        def helper(n, c, h, w, dtype=torch.float32):

            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps')

            ref_y = torch.max(cpu_x)
            y = torch.max(x)
            self.assertEqual(y, ref_y)

            for dim in [0, 1, 2, 3]:
                for keepdim in [True, False]:
                    y, idx = torch.max(x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.max(cpu_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

            y_0 = torch.ones(c, h, w, device='mps', dtype=dtype)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.max(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=dtype)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.max(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1 = torch.ones(n, h, w, device='mps', dtype=dtype)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.max(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=dtype)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.max(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2 = torch.ones(n, c, w, device='mps', dtype=dtype)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.max(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=dtype)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.max(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3 = torch.ones(n, c, h, device='mps', dtype=dtype)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.max(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=dtype)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.max(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5, torch.float32)
        helper(2, 8, 4, 5, torch.int32)
        # helper(2, 8, 4, 5, torch.int64)

    def test_median(self):
        def helper_dtype_int32(n1, n2, n3):
            cpu_x = torch.randint(50, (n1, n2, n3), device='cpu', dtype=torch.int32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)

            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        def helper_dtype_float32(n1, n2, n3):
            cpu_x = torch.randn(n1, n2, n3, device='cpu', dtype=torch.float32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)

            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        helper_dtype_int32(10, 10, 10)  # median at even place
        helper_dtype_int32(3, 3, 3)  # median at odd place
        helper_dtype_int32(1, 1, 1)
        helper_dtype_int32(1, 2, 3)
        helper_dtype_float32(10, 10, 10)
        helper_dtype_float32(3, 3, 3)
        helper_dtype_float32(1, 1, 1)

    def test_any(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.any(x)
                ref_y = torch.any(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.any(x, dim=0)
                refy_0 = torch.any(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.any(x, dim=1)
                refy_1 = torch.any(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.any(x, dim=1, keepdim=True)
                refy_1dim = torch.any(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)

                if (len(shape) > 2):
                    y_2 = torch.any(x, dim=2)
                    refy_2 = torch.any(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.any(x, dim=2, keepdim=True)
                    refy_2dim = torch.any(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.any(x, dim=3)
                    refy_3 = torch.any(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.any(x, dim=3, keepdim=True)
                    refy_3dim = torch.any(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)
        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))

    @unittest.skip("Test is crashing")
    def test_reduction_ops_5D(self):
        def helper(fn, dim):
            x_cpu = fn(torch.zeros(1, 1, 1, 1, 1), dim=dim)
            x_mps = fn(torch.zeros(1, 1, 1, 1, 1, device="mps"), dim=dim)
            self.assertEqual(x_cpu, x_mps.to('cpu'))
        for fn in [torch.any]:
            for dim in range(0, 4):
                helper(fn, dim)

    def test_all(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.all(x)
                ref_y = torch.all(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.all(x, dim=0)
                refy_0 = torch.all(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.all(x, dim=1)
                refy_1 = torch.all(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.all(x, dim=1, keepdim=True)
                refy_1dim = torch.all(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)
                if (len(shape) > 2):
                    y_2 = torch.all(x, dim=2)
                    refy_2 = torch.all(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.all(x, dim=2, keepdim=True)
                    refy_2dim = torch.all(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.all(x, dim=3)
                    refy_3 = torch.all(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.all(x, dim=3, keepdim=True)
                    refy_3dim = torch.all(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)

        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))
        x_cpu = torch.tensor([], dtype=torch.bool)
        x_mps = x_cpu.to("mps")
        assert x_cpu.all() == x_mps.all().cpu()

    # Test forward min
    def test_min_el(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            y = torch.min(x)
            ref_y = torch.min(cpu_x)
            self.assertEqual(y, ref_y)

            y_0, idx_0 = torch.min(x, dim=0)
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0 = torch.ones(c, h, w, device='mps', dtype=torch.float)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim, idx_0dim = torch.min(x, dim=0, keepdim=True)
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.float)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1, idx_1 = torch.min(x, dim=1)
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1 = torch.ones(n, h, w, device='mps', dtype=torch.float)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim, idx_1dim = torch.min(x, dim=1, keepdim=True)
            refy_1dim, refidx_1dim = torch.min(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.float)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.min(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2, idx_2 = torch.min(x, dim=2)
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2 = torch.ones(n, c, w, device='mps', dtype=torch.float)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim, idx_2dim = torch.min(x, dim=2, keepdim=True)
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.float)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3, idx_3 = torch.min(x, dim=3)
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3 = torch.ones(n, c, h, device='mps', dtype=torch.float)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim, idx_3dim = torch.min(x, dim=3, keepdim=True)
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.float)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5)

    # Test forward sum
    def test_sum(self):
        def helper(n, c, h, w, dtype=torch.float32):
            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            self.assertEqual(all_sum, all_sum_cpu)

            nil_dim_sum = torch.sum(x, dim=[])
            nil_dim_sum_cpu = torch.sum(cpu_x, dim=[])

            self.assertEqual(nil_dim_sum, nil_dim_sum_cpu)

            nil_dim_sum_keepdim = torch.sum(x, dim=[], keepdim=True)
            nil_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_sum_keepdim, nil_dim_sum_cpu_keepdim)

            zero_dim_sum = torch.sum(x, dim=[0])
            zero_dim_sum_cpu = torch.sum(cpu_x, dim=[0])

            self.assertEqual(zero_dim_sum, zero_dim_sum_cpu)

            zero_dim_sum_keepdim = torch.sum(x, dim=[0], keepdim=True)
            zero_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_sum_keepdim, zero_dim_sum_cpu_keepdim)

            zero_one_dim_sum = torch.sum(x, dim=[0, 1])
            zero_one_dim_sum_cpu = torch.sum(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_sum, zero_one_dim_sum_cpu)

            zero_one_dim_sum_keepdim = torch.sum(x, dim=[0, 1], keepdim=True)
            zero_one_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_sum_keepdim, zero_one_dim_sum_cpu_keepdim)

            two_three_dim_sum = torch.sum(x, dim=[2, 3])
            two_three_dim_sum_cpu = torch.sum(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_sum, two_three_dim_sum_cpu)

            two_three_keepdim_sum = torch.sum(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepsum_cpu = torch.sum(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_sum, two_three_dim_keepsum_cpu)

        helper(2, 8, 4, 5)
        helper(2, 8, 4, 5, dtype=torch.int32)
        helper(2, 8, 4, 5, dtype=torch.int64)
        helper(2, 8, 4, 5, dtype=torch.bool)

    # Test forward prod
    def test_prod(self):
        def helper(shape, dtype=torch.float32):
            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(1, 6, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_prod = torch.prod(x)
            all_prod_cpu = torch.prod(cpu_x)

            self.assertEqual(all_prod, all_prod_cpu)

            for dim in range(len(shape)):
                dim_prod = torch.prod(x, dim=dim)
                dim_prod_cpu = torch.prod(cpu_x, dim=dim)

                self.assertEqual(dim_prod, dim_prod_cpu)

                dim_prod_keepdim = torch.prod(x, dim=dim, keepdim=True)
                dim_prod_cpu_keepdim = torch.prod(cpu_x, dim=dim, keepdim=True)

                self.assertEqual(dim_prod_keepdim, dim_prod_cpu_keepdim)

        for dtype in [torch.float32, torch.int32, torch.int64, torch.bool]:
            helper((2, 3), dtype)

    # Test forward mean
    def test_mean(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_mean = torch.mean(x)
            all_mean_cpu = torch.mean(cpu_x)

            self.assertEqual(all_mean, all_mean_cpu)

            nil_dim_mean = torch.mean(x, dim=[])
            nil_dim_mean_cpu = torch.mean(cpu_x, dim=[])

            self.assertEqual(nil_dim_mean, nil_dim_mean_cpu)

            nil_dim_mean_keepdim = torch.mean(x, dim=[], keepdim=True)
            nil_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_mean_keepdim, nil_dim_mean_cpu_keepdim)

            zero_dim_mean = torch.mean(x, dim=[0])
            zero_dim_mean_cpu = torch.mean(cpu_x, dim=[0])

            self.assertEqual(zero_dim_mean, zero_dim_mean_cpu)

            zero_dim_mean_keepdim = torch.mean(x, dim=[0], keepdim=True)
            zero_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_mean_keepdim, zero_dim_mean_cpu_keepdim)

            zero_one_dim_mean = torch.mean(x, dim=[0, 1])
            zero_one_dim_mean_cpu = torch.mean(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_mean, zero_one_dim_mean_cpu)

            zero_one_dim_mean_keepdim = torch.mean(x, dim=[0, 1], keepdim=True)
            zero_one_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_mean_keepdim, zero_one_dim_mean_cpu_keepdim)

            two_three_dim_mean = torch.mean(x, dim=[2, 3])
            two_three_dim_mean_cpu = torch.mean(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_mean, two_three_dim_mean_cpu)

            two_three_keepdim_mean = torch.mean(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepmean_cpu = torch.mean(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_mean, two_three_dim_keepmean_cpu)

        helper(2, 8, 4, 5)

    # Test std
    def test_std(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            all_std = torch.std(x, unbiased=False)
            all_std_cpu = torch.std(cpu_x, unbiased=False)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=False)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=False)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=False)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=False)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=False)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=False)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=False)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=False)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=False)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=False)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=False)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=False)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=False)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=False)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=False)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=False)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

            all_std = torch.std(x, unbiased=True)
            all_std_cpu = torch.std(cpu_x, unbiased=True)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=True)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=True)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=True)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=True)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=True)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=True)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=True)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=True)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=True)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=True)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=True)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=True)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=True)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=True)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=True)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=True)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

        helper((4, 5, 6, 7))
        # verify if a change in shape of input would cause problems with graph caching
        helper((9, 5, 6, 7))

    # Test var
    def test_var_simple(self):
        def helper():

            shape = [2, 3, 4, 5]

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            for unbiased in [False, True]:
                for keepdim in [False, True]:

                    zero_dim_var = x.var(-1, keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = cpu_x.var(-1, keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    all_var = torch.var(x, unbiased=unbiased)
                    all_var_cpu = torch.var(cpu_x, unbiased=unbiased)

                    self.assertEqual(all_var, all_var_cpu)

                    nil_dim_var = torch.var(x, dim=[], keepdim=keepdim, unbiased=unbiased)
                    nil_dim_var_cpu = torch.var(cpu_x, dim=[], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(nil_dim_var, nil_dim_var_cpu)

                    zero_dim_var = torch.var(x, dim=[0], keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = torch.var(cpu_x, dim=[0], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    zero_one_dim_var = torch.var(x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)
                    zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

                    two_three_dim_var = torch.var(x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)
                    two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

        helper()

    # Test forward amax
    def test_amax(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amax(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amax(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test forward amin
    def test_amin(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amin(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amin(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test minimum and maximum
    def test_minimum_maximum(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')

            minimum_result_cpu = torch.minimum(cpu_x, cpu_y)
            minimum_result_mps = torch.minimum(mps_x, mps_y)
            self.assertEqual(minimum_result_cpu, minimum_result_mps)

            maximum_result_cpu = torch.maximum(cpu_x, cpu_y)
            maximum_result_mps = torch.maximum(mps_x, mps_y)
            self.assertEqual(maximum_result_cpu, maximum_result_mps)

        helper(1, 1, 4, 5)

    def test_clamp_fp16_fp32(self):
        cpu_x = torch.randn(10, device='cpu', dtype=torch.float, requires_grad=False)
        x = cpu_x.detach().clone().to('mps')

        dtype = torch.float16

        clamp_min_vals_mps = torch.ones(10, device="mps").to(torch.float16)
        clamp_max_vals_mps = torch.ones(10, device="mps").to(torch.float16) * 10
        clamp_result_mps = torch.clamp(x, clamp_min_vals_mps, clamp_max_vals_mps)

        clamp_min_vals_cpu = torch.ones(10, device="cpu").to(torch.float16)
        clamp_max_vals_cpu = torch.ones(10, device="cpu").to(torch.float16) * 10
        clamp_result_cpu = torch.clamp(cpu_x, clamp_min_vals_cpu, clamp_max_vals_cpu)

        self.assertEqual(clamp_result_mps, clamp_result_cpu)

    # Test clamp_min
    def test_clamp_min(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_min_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            clamp_min_result = torch.clamp_min(x, min=5.0)
            clamp_min_result_cpu = torch.clamp_min(cpu_x, min=5.0)

            self.assertEqual(clamp_min_result, clamp_min_result_cpu)

            clamp_min_t_result = torch.clamp_min(x, min=min_t)
            clamp_min_t_result_cpu = torch.clamp_min(cpu_x, min=cpu_min_t)

            self.assertEqual(clamp_min_t_result, clamp_min_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp_max

    def test_clamp_max(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_max_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            clamp_max_result = torch.clamp_max(x, max=100.0)
            clamp_max_result_cpu = torch.clamp_max(cpu_x, max=100.0)

            self.assertEqual(clamp_max_result, clamp_max_result_cpu)

            clamp_max_t_result = torch.clamp_max(x, max=max_t)
            clamp_max_t_result_cpu = torch.clamp_max(cpu_x, max=cpu_max_t)

            self.assertEqual(clamp_max_t_result, clamp_max_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp
    def test_clamp(self):
        def helper(n, c, h, w):
            import numpy as np
            upper_bound = 1000
            half_upper_bound = upper_bound / 2

            # x=[0..1000)
            x_arr = upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_x = torch.tensor(x_arr, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # x=[0..500)
            min_arr = half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_min_t = torch.tensor(min_arr, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            # x=[500..1000), to ensure max's are greater than mins
            max_arr = (half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)) + half_upper_bound
            cpu_max_t = torch.tensor(max_arr, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            # [200..600]: just an arbitrary range between [0..1000]
            clamp_result = torch.clamp(x, min=200.0, max=600.0)
            clamp_result_cpu = torch.clamp(cpu_x, min=200.0, max=600.0)
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test optional scalar refs and cached graph keys by passing only max
            clamp_opt_result = torch.clamp(x, max=600.0)
            clamp_opt_result_cpu = torch.clamp(cpu_x, max=600.0)
            self.assertEqual(clamp_opt_result, clamp_opt_result_cpu)

            clamp_t_result = torch.clamp(x, min=min_t, max=max_t)
            clamp_t_result_cpu = torch.clamp(cpu_x, min=cpu_min_t, max=cpu_max_t)
            self.assertEqual(clamp_t_result, clamp_t_result_cpu)

            # test optional tensor refs and cached graph keys by passing only max
            clamp_topt_result = torch.clamp(x, max=max_t)
            clamp_topt_result_cpu = torch.clamp(cpu_x, max=cpu_max_t)
            self.assertEqual(clamp_topt_result, clamp_topt_result_cpu)

            # test strided x
            clamp_result = torch.clamp(x.movedim(0, -1), min=200.0, max=600.0)
            clamp_result_cpu = torch.clamp(cpu_x.movedim(0, -1), min=200.0, max=600.0)
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test strided x, min_t, max_t
            clamp_result = torch.clamp(x.movedim(0, -1), min=min_t.movedim(0, -1), max=max_t.movedim(0, -1))
            clamp_result_cpu = torch.clamp(cpu_x.movedim(0, -1), min=cpu_min_t.movedim(0, -1), max=cpu_max_t.movedim(0, -1))
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test strided min_t, max_t
            clamp_result = torch.clamp(
                x.movedim(0, -1).clone(memory_format=torch.contiguous_format),
                min=min_t.movedim(0, -1),
                max=max_t.movedim(0, -1)
            )
            clamp_result_cpu = torch.clamp(
                cpu_x.movedim(0, -1).clone(memory_format=torch.contiguous_format),
                min=cpu_min_t.movedim(0, -1),
                max=cpu_max_t.movedim(0, -1)
            )
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test inplace clamping
            x.clamp_(min=200.0, max=600.0)
            cpu_x.clamp_(min=200.0, max=600.0)
            self.assertEqual(cpu_x, x)

        helper(2, 8, 4, 5)

    def test_divmode(self):
        def helper(shape, rounding_mode):
            for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
                if ((rounding_mode is not None and "floor" in rounding_mode and dtype == torch.int64) or
                        (rounding_mode is not None and "trunc" in rounding_mode and dtype == torch.float16)) is False:
                    cpu_x = None
                    cpu_y = None
                    if (dtype in [torch.float32, torch.float16]):
                        cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                    else:
                        cpu_x = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)

                    mps_x = cpu_x.detach().clone().to('mps')
                    # clamp to avoid division by 0
                    mps_y = cpu_y.detach().clone().to('mps')

                    if (rounding_mode == "floor_divide"):
                        result_div_cpu = torch.floor_divide(cpu_x, cpu_y)
                        result_div_mps = torch.floor_divide(mps_x, mps_y)
                        self.assertEqual(result_div_mps, result_div_cpu)
                    else:
                        result_div_cpu = torch.div(cpu_x, cpu_y, rounding_mode=rounding_mode)
                        result_div_mps = torch.div(mps_x, mps_y, rounding_mode=rounding_mode)
                        self.assertEqual(result_div_mps, result_div_cpu)

        helper((2, 8, 4, 5), None)
        helper((2, 8, 4, 5), "floor")
        helper((2, 8, 4, 5), "trunc")
        helper((2, 8, 4, 5), "floor_divide")

    def test_rounding(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')

            result_floor_cpu = torch.floor(cpu_x)
            result_floor_mps = torch.floor(mps_x)
            self.assertEqual(result_floor_mps, result_floor_cpu)

            result_ceil_cpu = torch.ceil(cpu_x)
            result_ceil_mps = torch.ceil(mps_x)
            self.assertEqual(result_ceil_mps, result_ceil_cpu)

            result_trunc_cpu = torch.trunc(cpu_x)
            result_trunc_mps = torch.trunc(mps_x)
            self.assertEqual(result_trunc_mps, result_trunc_cpu)

            result_round_cpu = torch.round(cpu_x)
            result_round_mps = torch.round(mps_x)
            self.assertEqual(result_round_mps, result_round_cpu)

        helper((2, 6, 3, 5))
        helper((2, 8, 4, 5))

    def test_remainder(self):
        res_cpu = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="cpu"), torch.tensor(2, device="cpu", dtype=torch.int32))
        res_mps = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="mps"), torch.tensor(2, device="mps", dtype=torch.int32))
        self.assertEqual(res_cpu, res_mps)

        res_cpu = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cpu"), -1.5)
        res_mps = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="mps"), -1.5)
        self.assertEqual(res_cpu, res_mps)

    def test_expand(self):
        def helper(n, c):
            values = [[1.0], [4.0], [7.0]]
            cpu_x = torch.tensor(values, device='cpu')
            x = cpu_x.detach().clone().to('mps')

            strided_cpu = torch.as_strided(cpu_x, (3, 4), (1, 0))
            strided_mps = torch.as_strided(x, (3, 4), (1, 0))

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 1)

    def test_im2col(self):
        def helper(x):
            return torch.nn.functional.unfold(x, kernel_size=(10, 15), dilation=2, padding=5, stride=3)
        x_cpu = torch.rand(1, 1, 200, 100)
        x = x_cpu.detach().clone().to('mps')
        self.assertEqual(helper(x_cpu), helper(x))

    def test_select(self):
        def helper(n, c):
            cpu_x = torch.randn(n, c, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1))
            strided_mps = torch.as_strided(x, (3, 1), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (1, 3), (3, 1))
            strided_mps = torch.as_strided(x, (1, 3), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1), storage_offset=1)
            strided_mps = torch.as_strided(x, (3, 1), (3, 1), storage_offset=1)

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 3)

    def test_topk(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            for largest_val in [True, False]:
                if (type(shape) == tuple):
                    for curr_dim in range(0, len(shape)):
                        dim_size = shape[curr_dim]
                        for k in range(1, dim_size + 1):
                            topk_values, topk_indices = torch.topk(x, k, dim=curr_dim, largest=largest_val)
                            topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=curr_dim, largest=largest_val)
                            self.assertEqual(topk_values, topk_values_cpu)
                            self.assertEqual(topk_indices, topk_indices_cpu)
                else:
                    for k in range(1, shape):
                        topk_values, topk_indices = torch.topk(x, k, dim=0, largest=largest_val)
                        topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=0, largest=largest_val)
                        self.assertEqual(topk_values, topk_values_cpu)
                        self.assertEqual(topk_indices, topk_indices_cpu)

        helper(2)
        helper((5, 1))
        helper((1, 5))
        helper((5, 9, 7, 4))
        helper((50, 20, 7, 4))

    def test_sort(self):
        for SIZE in (4, 2049):
            device = 'mps'
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

    def test_upsample_nearest2d(self):
        def helper(N, C, H, W, memory_format):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W).to(memory_format=memory_format)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_nearest2d = nn.UpsamplingNearest2d(scale_factor=(i, j))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)
                    upsample_nearest2d = nn.UpsamplingNearest2d((i * H, j * W))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        for memory_format in [torch.channels_last, torch.contiguous_format]:
            helper(1, 1, 4, 4, memory_format=memory_format)
            helper(7, 5, 3, 2, memory_format=memory_format)

    def test_upsample_bilinear2d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=(i, j))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    upsample_bilinear2d = nn.UpsamplingBilinear2d((i * H, j * W))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    def test_interpolate(self):
        def helper(shape, output_size, scales, mode, align_corners=False):
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            # align_corners is used for 2D interpolation only
            if (align_corners is True and len(shape) > 3 and mode == 'bilinear'):
                if scales is not None:
                    outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode, align_corners=align_corners)
                else:
                    outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode, align_corners=align_corners)
            elif scales is not None:
                outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode)
            else:
                outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode)

            self.assertEqual(outputCPU, outputMPS)

            # backward pass (chose 0.6 just to have the grad_output != 1)
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 1D interpolation
        for mode in ['nearest', 'nearest-exact']:
            helper([2, 3, 4], [3], None, mode)  # downsample with size
            helper([2, 3, 4], [6], None, mode)  # upsample with size
            helper([2, 3, 4], None, [0.6], mode)  # downsample with scale factor
            helper([2, 3, 4], None, [1.7], mode)  # upsample with scale factor
        # 2D interpolation
        for mode in ['nearest', 'nearest-exact', 'bilinear']:
            helper([2, 3, 4, 5], [3, 4], None, mode)  # downsample_nearest with size
            helper([2, 3, 4, 5], [6, 7], None, mode)  # upsample_nearest with size
            helper([2, 3, 4, 5], None, [0.6, 0.7], mode)  # downsample_nearest with scale factor
            helper([2, 3, 4, 5], None, [1.4, 1.7], mode)  # upsample_nearest with scale factor
        # align_corners=True
        helper([2, 3, 4, 5], [3, 4], None, 'bilinear', True)
        helper([2, 3, 4, 5], None, [1.4, 1.7], 'bilinear', True)

    # Test concat forward
    def test_cat1(self):
        def helper(shape_x, shape_y, shape_z):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cat = torch.cat([x, y, z], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(cat, cat_cpu)

        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        helper([2, 2, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([0, 2, 4, 5], [0, 3, 4, 5], [0, 5, 4, 5])
        helper([2, 2, 6, 5], [0], [2, 5, 6, 5])
        helper([0], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 3, 4, 5], [2, 5, 4, 5], [0])
        helper([2, 2, 6, 5], [2, 0, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 0, 6, 5])

    def test_constant_pad(self):
        m = torch.nn.ConstantPad2d((-2, -2, -2, -2), 3.5)
        input_cpu = torch.randn(1, 16, 16, 16)
        input_mps = input_cpu.detach().clone().to("mps")
        r_cpu = m(input_cpu)
        r_mps = m(input_mps)
        self.assertEqual(r_cpu, r_mps.to("cpu"))

        # Arbitrary input dimensions
        pad = (1, 1, 0, 0, 0, 0)
        value = 3.5
        input_cpu = torch.randn((1, 1, 3, 3, 3, 3, 3, 3, 3, 3))
        input_mps = input_cpu.detach().clone().to("mps")
        r_cpu = F.pad(input_cpu, pad=pad, value=value)
        r_mps = F.pad(input_mps, pad=pad, value=value)
        self.assertEqual(r_cpu, r_mps.to("cpu"))

    def test_circular_pad(self):
        # https://github.com/pytorch/pytorch/issues/80856
        k_cpu = torch.ones(3, 3, 9, 9)
        k_mps = k_cpu.detach().clone().to("mps")

        x_cpu = torch.rand(1, 3, 32, 32)
        x_mps = x_cpu.detach().clone().to("mps")

        x_pad_cpu = F.pad(x_cpu, (2, 2, 2, 2), mode='circular')
        x_pad_mps = F.pad(x_mps, (2, 2, 2, 2), mode='circular')

        y_cpu = F.conv2d(x_pad_cpu, k_cpu)
        y_mps = F.conv2d(x_pad_mps, k_mps)

        self.assertEqual(y_cpu, y_mps.cpu())

    def test_constant_pad_4d_warning(self):
        inputCPU = torch.rand((1, 2, 2, 2, 1, 1))
        inputMPS = inputCPU.detach().clone().to('mps')
        outputCPU = F.pad(inputCPU, [0, 0, 0, 0, 0, 0, 1, 0])
        outputMPS = F.pad(inputMPS, [0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(outputCPU, outputMPS)

    def test_pad(self):
        def helper(shape, padding, op, value=0):
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            if (op in [nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]):
                padCriteria = op(padding, value)
            else:
                padCriteria = op(padding)
            outputCPU = padCriteria(inputCPU)
            outputMPS = padCriteria(inputMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass (chose 0.6 just to have the grad_output != 1)
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 1D Padding
        helper((2, 4, 3), 2, nn.ReflectionPad1d)
        # verify if a change in shape of input would cause problems with graph caching
        helper((2, 4, 4), (1, 3), nn.ReflectionPad1d)
        # Replication 1D
        helper((2, 1, 6), 3, nn.ReplicationPad1d)
        # Constant Pad 1D
        helper((2, 3, 4), 2, nn.ConstantPad1d)
        # Constant Pad 1D with single dimension input
        helper((16), (1, 2), nn.ConstantPad1d)

        # 2D Padding
        helper((1, 2, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # verify if a change in shape of input would cause problems with graph caching
        helper((2, 4, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # this should make the padding (2, 2, 2, 2)
        helper((2, 1, 6, 8), 2, nn.ReplicationPad2d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 1, 6, 8), (2, 4, 3, 5), nn.ReplicationPad2d)
        # Constant Pad 2D
        helper((2, 1, 6, 8), (2, 4, 3, 5), nn.ConstantPad2d)
        # input size < pad size
        helper((1, 2, 3), (0, 0, 0, 1), nn.ConstantPad2d)
        # pad dims < input dims
        helper((50, 9, 300), (0, 0, 0, 31), nn.ConstantPad2d)
        # pad dims == input dims
        helper((1, 3), (0, 2, 0, 1), nn.ConstantPad2d)
        # input.numel() == 0 but output.numel() > 0
        helper((0, 3, 3), (1, 1, 1, 1, 1, 1), nn.ConstantPad2d)
        # pad dims < input dims - 2
        helper((1, 2, 3, 4), (1, 2), nn.ConstantPad2d)

        # 3D Padding
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReflectionPad3d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReplicationPad3d)
        # case where input_d == pad_front/back for ReplicationPad3d
        helper((3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6), nn.ReplicationPad3d)
        # Constant Pad 3D
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ConstantPad3d)
        # input size < pad size
        helper((2, 4, 6), (1, 3, 3, 5, 3, 4), nn.ConstantPad3d)
        # check the workaround for the right padding bug in Monterey
        helper((1, 2, 2, 2, 2), (0, 1), nn.ConstantPad3d)

    # Test stack forward
    def test_stack(self):
        # All shapes must be same
        def helper(shape, dtype=torch.float32):

            x, cpu_x = None, None
            y, cpu_y = None, None
            z, cpu_z = None, None

            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                y = cpu_y.detach().clone().to('mps').requires_grad_()
                cpu_z = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                z = cpu_z.detach().clone().to('mps').requires_grad_()

            stack = torch.stack([x, y, z], dim=1)
            stack_cpu = torch.stack([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(stack, stack_cpu)

        helper([2, 8, 4, 5])
        helper([2, 8, 4, 5], dtype=torch.float16)
        helper([2, 8, 4, 5], dtype=torch.int32)
        helper([2, 8, 4, 5], dtype=torch.int64)
        helper([2, 8, 4, 5], dtype=torch.bool)
        # Empty test - Currently failing! Empty tensor not handled!
        # helper([0, 2, 4, 5])

    # Test abs
    def test_abs(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            abs_result = torch.abs(x)
            abs_result_cpu = torch.abs(cpu_x)

            self.assertEqual(abs_result, abs_result_cpu)

        helper((2, 8, 4, 5))

    def test_log(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_result = torch.log(x)
            log_result_cpu = torch.log(cpu_x)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_log_ten(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_ten_result = torch.log10(x)
            log_ten_result_cpu = torch.log10(cpu_x)

            self.assertEqual(log_ten_result, log_ten_result_cpu)

        helper((2, 8, 4, 5))

    def test_log_two(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_two_result = torch.log2(x)
            log_two_result_cpu = torch.log2(cpu_x)

            self.assertEqual(log_two_result, log_two_result_cpu)

        helper((2, 8, 4, 5))

    def test_log1p(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_result = torch.log1p(x)
            log_result_cpu = torch.log1p(cpu_x)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_logaddexp(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            log_result = torch.logaddexp(x, y)
            log_result_cpu = torch.logaddexp(cpu_x, cpu_y)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_logaddexp2(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            log_result = torch.logaddexp2(x, y)
            log_result_cpu = torch.logaddexp2(cpu_x, cpu_y)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    # Test concat forward
    def test_cat2(self):

        def helper1(shape_x, shape_y, shape_z, shape_w):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cpu_w = torch.randn(shape_w, device='cpu', dtype=torch.float, requires_grad=False)
            w = cpu_w.detach().clone().to('mps')

            cat = torch.cat([x, y, z, w], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z, cpu_w], dim=1)

            self.assertEqual(cat, cat_cpu)

        def helper(shape_x, shape_y, shape_z):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cat = torch.cat([x, y, z], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(cat, cat_cpu)

        helper([2, 8, 4, 5], [2, 10, 4, 5], [2, 6, 4, 5])
        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        # Empty test - Currently failing! Empty tensor not handled!
        # helper([0, 2, 4, 5], [2, 0, 4, 5], [2, 5, 0, 5])

    # Test isnan
    def test_isnan(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            nan_index = [random.randrange(0, shape[0])]
            # make a selected row inf
            cpu_x.index_put_(indices=[torch.tensor(nan_index)], values=torch.tensor(float('nan')))
            x = cpu_x.detach().clone().to('mps')

            isnan_result = torch.isnan(x)
            isnan_result_cpu = torch.isnan(cpu_x)

            self.assertEqual(isnan_result, isnan_result_cpu)

        helper((8, 2, 4, 5))

    # Test reciprocal
    def test_reciprocal(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            reciprocal_result = torch.reciprocal(x)
            reciprocal_result_cpu = torch.reciprocal(cpu_x)

            cpu_grad = torch.ones_like(reciprocal_result_cpu)
            grad = cpu_grad.to('mps')

            reciprocal_result.backward(gradient=grad)
            reciprocal_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(reciprocal_result, reciprocal_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))

    # Test sqrt
    def test_sqrt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sqrt_result = torch.sqrt(x)
            sqrt_result_cpu = torch.sqrt(cpu_x)

            cpu_grad = torch.ones_like(sqrt_result_cpu)
            grad = cpu_grad.to('mps')

            sqrt_result.backward(gradient=grad)
            sqrt_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(sqrt_result, sqrt_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))

    # Test selu, elu, celu
    def test_elu(self):
        def helper(shape, alpha=1.0, memory_format=torch.contiguous_format):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_x = cpu_x.to(memory_format=memory_format).requires_grad_()

            x = cpu_x.detach().clone().to('mps').requires_grad_(True)
            for activation_func in [torch.nn.ELU(alpha=alpha), torch.nn.CELU(alpha=alpha), torch.nn.SELU()]:
                elu_result = activation_func(x)
                elu_result_cpu = activation_func(cpu_x)

                cpu_grad = torch.randn(elu_result_cpu.shape)
                grad = cpu_grad.to('mps')

                elu_result.backward(gradient=grad)
                elu_result_cpu.backward(gradient=cpu_grad)

                self.assertEqual(elu_result, elu_result_cpu)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for memory_fromat in [torch.channels_last, torch.contiguous_format]:
            for shape in [(2, 8, 4, 5)]:
                for alpha in [0.000001, 1.0, 2.3, 0.34, 23]:
                    helper(shape, alpha, memory_fromat)

    # Test glu
    def test_glu(self):
        def helper(shape, dim=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            for activation_func in [torch.nn.GLU(dim=dim)]:
                glu_result = activation_func(x)
                glu_result_cpu = activation_func(cpu_x)

                cpu_grad = torch.randn(glu_result_cpu.shape)
                grad = cpu_grad.to('mps')

                glu_result.backward(gradient=grad)
                glu_result_cpu.backward(gradient=cpu_grad)

                self.assertEqual(glu_result, glu_result_cpu)
                self.assertEqual(x.grad, cpu_x.grad)

        for shape in [[4], (2, 4), (2, 8, 4, 6)]:
            for dim in range(len(shape)):
                helper(shape, dim)

    # Test softplus
    def test_softplus(self):
        def helper(shape, beta, threshold, dtype):
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softplus_result = torch.nn.Softplus(beta=beta, threshold=threshold)(x)
            softplus_result_cpu = torch.nn.Softplus(beta=beta, threshold=threshold)(cpu_x)

            cpu_grad = torch.randn(softplus_result.shape)
            grad = cpu_grad.to('mps')

            softplus_result.backward(gradient=grad)
            softplus_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softplus_result, softplus_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape, beta, threshold, dtype in product(
            [(), (2, 3), (10, 10), (2, 3, 4, 5)],
            [0.5, 1, 2, 3, 4],
            [0.5, 20, 30, 40, 50],
            [torch.float16, torch.float32]
        ):
            helper(shape, beta, threshold, dtype)

    # Test silu

    def test_silu(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            silu_result = torch.nn.SiLU()(x)
            silu_result_cpu = torch.nn.SiLU()(cpu_x)

            cpu_grad = torch.randn(silu_result_cpu.shape)
            grad = cpu_grad.to('mps')

            silu_result.backward(gradient=grad)
            silu_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(silu_result, silu_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [[], (2, 3), (2, 8, 4, 5)]:
            helper(shape)

    def test_cast_mps_to_cpu(self):
        def helper(src_dtype, dst_dtype):
            input = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            input_cast_mps = input.to('mps')
            input_cast_cpu = input_cast_mps.to('cpu', dtype=dst_dtype)

            # needs to match the initial Tensor
            self.assertEqual(input_cast_cpu, input.to(dtype=dst_dtype))
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)

    def test_cast_mps_to_mps(self):
        def helper(src_dtype, dst_dtype):
            input_cpu = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            input_mps = input_cpu.to('mps')
            output_mps = input_mps.to(dtype=dst_dtype)
            output_cpu = input_cpu.to(dtype=dst_dtype)
            self.assertEqual(output_mps.cpu(), output_cpu)
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)
        helper(torch.half, torch.long)
        helper(torch.float, torch.int)

    def test_avg_pool2d_count_include_pad(self):
        cpu_x = torch.randn((1, 3, 9, 9), device='cpu', dtype=torch.float, requires_grad=True)
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        pool = torch.nn.AvgPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), ceil_mode=True, count_include_pad=True)
        ref_y = pool(cpu_x)
        y = pool(x)
        self.assertEqual(y, ref_y)
        cpu_grad = torch.randn(ref_y.shape)
        grad = cpu_grad.to('mps')
        ref_y.backward(gradient=cpu_grad)
        y.backward(gradient=grad)
        self.assertEqual(x.grad, cpu_x.grad)

    # Test adaptive avg pool2d - when the input size is a multiple of output size
    # Not testing for channels last right now
    def test_adaptive_avg_pool2d_simple(self):
        def helper(input_shape, out_shape, channels_last):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            avg_result = torch.nn.AdaptiveAvgPool2d(out_shape)(x)
            avg_result_cpu = torch.nn.AdaptiveAvgPool2d(out_shape)(cpu_x)

            cpu_grad = torch.randn(avg_result_cpu.shape)
            grad = cpu_grad.to('mps')

            avg_result.backward(gradient=grad)
            avg_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(avg_result, avg_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 2, 4, 4), (2, 2), False)
        helper((2, 2, 9, 9), (3, 3), False)
        helper((2, 2, 9, 9), (9, 9), False)
        helper((2, 2, 16, 16), (2, 2), False)
        helper((2, 2, 16, 16), (2, 16), False)

        helper((2, 16, 16), (4, 4), False)

        # Output shape larger than input shape

        helper((2, 2, 4, 4), (8, 8), False)
        helper((2, 2, 2, 2), (4, 4), False)
        helper((2, 2, 3, 3), (9, 9), False)
        helper((2, 2, 2, 2), (16, 16), False)
        helper((2, 2, 2, 16), (16, 16), False)

        helper((2, 4, 4), (16, 16), False)

        try:
            helper((2, 2, 3, 3), (7, 7), False)
        except Exception as e:
            pass

    # Test max avg pool2d - when the input size is a multiple of output size
    # Not testing for channels last right now
    def test_adaptive_max_pool2d_simple(self):
        def helper(input_shape, out_shape, return_indices, dtype, channels_last=False):
            cpu_x = None
            if (dtype in [torch.float16, torch.float32]):
                cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            else:
                cpu_x = torch.randint(50, input_shape, device='cpu', dtype=dtype, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            max_result, max_indices = None, None
            max_result_cpu, max_indices_cpu = None, None

            if (return_indices):
                max_result, max_indices = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu, max_indices_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)
            else:
                max_result = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)

            cpu_grad = torch.randn(max_result_cpu.shape)
            grad = cpu_grad.to('mps')

            max_result.backward(gradient=grad)
            max_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(max_result, max_result_cpu)
            if (return_indices):
                self.assertEqual(max_indices, max_indices_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dtype in [torch.float32]:
            for return_indices in [False, True]:
                helper((2, 2, 4, 4), (2, 2), return_indices, dtype)
                helper((2, 2, 9, 9), (3, 3), return_indices, dtype)
                helper((2, 2, 9, 9), (9, 9), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 2), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 16), return_indices, dtype)
                helper((2, 16, 16), (4, 4), return_indices, dtype)

    def test_gelu_simple(self):
        def helper(shape, dtype=torch.float):
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            gelu_result = torch.nn.GELU()(x)
            # GELU is not supported on CPU, so cast it to float
            gelu_result_cpu = torch.nn.GELU()(cpu_x.to(torch.float))

            cpu_grad = torch.ones_like(gelu_result_cpu)
            grad = cpu_grad.to('mps')

            gelu_result.backward(gradient=grad)
            gelu_result_cpu.backward(gradient=cpu_grad)

            atol = 1e-5 if dtype == torch.float else 1e-2
            rtol = 1e-3 if dtype == torch.float else 1e-2
            self.assertEqual(gelu_result, gelu_result_cpu.to(dtype), atol=atol, rtol=rtol)
            self.assertEqual(x.grad, cpu_x.grad, atol=atol, rtol=rtol)

        # Test empty shape too
        for dtype in [torch.float, torch.half]:
            for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
                helper(shape, dtype)
        # Test that gelu would raise an assert for integral types
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            self.assertRaises(RuntimeError, lambda: torch.nn.GELU()(torch.randint(100, (2,), dtype=dtype, device="mps")))

    def test_gelu(self):
        def _test_gelu(n, m, dtype, contiguous, atol=None, rtol=None):
            numpy_dtype = {
                torch.bfloat16: torch.float, torch.float: torch.float, torch.double: torch.double
            }[dtype]
            devices = ['cpu']
            devices += ['mps']

            def _gelu_ref(X):
                return X * stats.norm.cdf(X)

            for d in devices:
                X = torch.rand(n, m, dtype=dtype, requires_grad=True, device=d)[:, ::2]
                res = X
                ref = (X.to(numpy_dtype).cpu().detach().numpy())
                self.assertEqual(res, ref, rtol=rtol, atol=atol, exact_dtype=False)

        for n in [1, 5, 10]:
            for m in [1, 5, 10]:
                _test_gelu(n, m, torch.float32, True)
                _test_gelu(n, m, torch.float32, False)

        # Test multi threaded
        num_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            _test_gelu(32, 32, torch.float32, False)
        finally:
            torch.set_num_threads(num_threads)

    def test_gelu_tanh(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            x = cpu_x.detach().clone().to('mps')

            gelu_tanh_result = torch.nn.functional.gelu(x, approximate='tanh')
            gelu_tanh_result_cpu = torch.nn.functional.gelu(cpu_x, approximate='tanh')
            self.assertEqual(gelu_tanh_result, gelu_tanh_result_cpu)

        helper((2, 8, 4, 5))

    # Test hardtanh
    def test_hardtanh(self):
        def helper(shape, min_val, max_val, inplace=False):
            cpu_x = None
            x = None

            if (not inplace):
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')

            hardtanh_result = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(x)
            hardtanh_result_cpu = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(cpu_x)

            self.assertEqual(hardtanh_result, hardtanh_result_cpu)

            if (not inplace):
                cpu_grad = torch.randn(hardtanh_result_cpu.shape)
                grad = cpu_grad.to('mps')
                hardtanh_result.backward(gradient=grad)
                hardtanh_result_cpu.backward(gradient=cpu_grad)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            for min_val, max_val in zip([-1, -2, 3], [1, -1, 4]):
                helper(shape, min_val, max_val)
                helper(shape, min_val, max_val, inplace=True)

    def test_hardswish(self):
        def helper(shape, inplace=False, requires_grad=True):
            m = nn.Hardswish(inplace=inplace)

            input_cpu = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=requires_grad)
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            if inplace and requires_grad:  # check that both raise runtime error
                self.assertRaises(RuntimeError, lambda: m(input_cpu))
                self.assertRaises(RuntimeError, lambda: m(input_mps))
                return

            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            cpu_grad = torch.ones_like(output_cpu)
            mps_grad = cpu_grad.to('mps')

            self.assertEqual(output_cpu, output_mps)

            if requires_grad:
                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                self.assertEqual(input_cpu.grad, input_mps.grad)

        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            helper(shape, inplace=False, requires_grad=False)
            helper(shape, inplace=True, requires_grad=False)
            helper(shape, inplace=False, requires_grad=True)
            helper(shape, inplace=True, requires_grad=True)

    def test_transpose_2D(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        values1 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')
        mps_x1 = torch.tensor(values1, device='mps')

        cpu_transpose = torch.transpose(cpu_x, 0, 1)
        mps_transpose = torch.transpose(mps_x, 0, 1)
        self.assertEqual(cpu_transpose, mps_transpose.to('cpu'))

    def test_transpose_3D(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        self.assertEqual(cpu_transpose1, mps_transpose1)

        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        self.assertEqual(cpu_transpose2, mps_transpose2)

        cpu_transpose3 = torch.transpose(cpu_x, 1, 2)
        mps_transpose3 = torch.transpose(mps_x, 1, 2).to('cpu')
        self.assertEqual(cpu_transpose3, mps_transpose3)


    def test_transpose_4D(self):
        values = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                  [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        self.assertEqual(cpu_transpose1, mps_transpose1)

        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        self.assertEqual(cpu_transpose2, mps_transpose2)

        cpu_transpose3 = torch.transpose(cpu_x, 0, 3)
        mps_transpose3 = torch.transpose(mps_x, 0, 3).to('cpu')
        self.assertEqual(cpu_transpose3, mps_transpose3)

        cpu_transpose4 = torch.transpose(cpu_x, 3, 1)
        mps_transpose4 = torch.transpose(mps_x, 3, 1).to('cpu')
        self.assertEqual(cpu_transpose4, mps_transpose4)

        cpu_transpose5 = torch.transpose(cpu_x, 3, 2)
        mps_transpose5 = torch.transpose(mps_x, 3, 2).to('cpu')
        self.assertEqual(cpu_transpose5, mps_transpose5)

        cpu_transpose6 = torch.transpose(cpu_x, 1, 2)
        mps_transpose6 = torch.transpose(mps_x, 1, 2).to('cpu')
        self.assertEqual(cpu_transpose6, mps_transpose6)

    # Test sign
    def test_sign(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sign_result = torch.sign(x)
            sign_result_cpu = torch.sign(cpu_x)

            cpu_grad = torch.ones_like(sign_result_cpu)
            grad = cpu_grad.to('mps')

            sign_result.backward(gradient=grad)
            sign_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(sign_result, sign_result_cpu)

        helper((2, 8, 4, 5))

    def test_signbit(self):
        def helper(shape, dtype):
            cpu_x = torch.randn(shape, device='cpu').to(dtype)
            x = cpu_x.clone().to('mps')

            signbit_result = torch.signbit(x)
            signbit_result_cpu = torch.signbit(cpu_x)

            self.assertEqual(signbit_result, signbit_result_cpu)

        helper((2, 8, 4, 5), torch.int)
        helper((2, 8, 4, 5), torch.float)
        helper((2, 8, 4, 5), torch.int64)

    # Test neg
    def test_neg(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            neg_result = torch.neg(x)
            neg_result_cpu = torch.neg(cpu_x)

            cpu_grad = torch.ones_like(neg_result_cpu)
            grad = cpu_grad.to('mps')

            neg_result.backward(gradient=grad)
            neg_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(neg_result, neg_result_cpu)

        helper((2, 8, 4, 5))

    def test_neg_strided_input(self):
        # See https://github.com/pytorch/pytorch/issues/98074#issuecomment-1496088337
        x = torch.arange(18.0, device='mps').reshape(2, 3, 3)
        y = x.permute(1, 0, 2)[..., 1]
        z = y + y.neg()
        self.assertEqual(z.abs().max().item(), 0.0)

    # Test index add
    def test_index_add(self):
        def helper(shape, dim, index, source_shape, alpha, x_dtype=torch.float32, idx_dtype=torch.int32):
            cpu_x = torch.randn(shape, device='cpu', dtype=x_dtype, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            cpu_source = torch.randn(source_shape, device='cpu', dtype=x_dtype, requires_grad=False)
            source = cpu_source.detach().clone().to('mps')

            idx_result = torch.index_add(x, dim=dim, index=idx, source=source, alpha=alpha)
            idx_result_cpu = torch.index_add(cpu_x, dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha)
            self.assertEqual(idx_result, idx_result_cpu)

        helper((2, 8, 4, 5), 0, [0, 1, 0], (3, 8, 4, 5), 5)
        helper((8, 8, 4, 5), 0, [7], (1, 8, 4, 5), 6.0)
        helper((2, 8, 4, 5), 1, [0, 3, 7], (2, 3, 4, 5), 5)
        helper((2, 8, 4, 5), 2, [3, 0], (2, 8, 2, 5), 3.0)
        helper((2, 8, 4, 5), 3, [2, 3, 0], (2, 8, 4, 3), 4)
        helper((2, 3, 3), -1, [1, 2], (2, 3, 2), 6.0)
        # test result dim=1
        helper((2,), 0, [1], (1,), 6.0)
        helper(2, 0, 1, 1, 6)
        # test float16
        helper((2,), 0, [1], (1,), 6.0, x_dtype=torch.float16)

    # Test flip
    def test_flip(self):
        def helper(shape, dims):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            flip_result = torch.flip(x, dims=dims)
            flip_result_cpu = torch.flip(cpu_x, dims=dims)

            self.assertEqual(flip_result, flip_result_cpu)

        helper((2, 8, 4, 5), [0])
        helper((8, 8, 4, 5), [0, 1])
        helper((2, 8, 4, 5), (0, 1, 2, 3))
        helper((2, 3, 3), (-1,))
        # empty dims
        helper((2, 8, 4, 5), [])
        # input.numel() == 1
        helper((1,), (0,))
        # input.numel() == 0
        helper((0,), (0,))
        # none of dims that needs to be flipped
        helper((1, 3), [0])

    # Test index select
    def test_index_select(self):
        def helper(shape, dim, index, idx_dtype=torch.int32):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            idx_result = torch.index_select(x, dim=dim, index=idx)
            idx_result_cpu = torch.index_select(cpu_x, dim=dim, index=cpu_idx)

            self.assertEqual(idx_result, idx_result_cpu)

        helper((2, 8, 4, 5), 0, [1])
        helper((8, 8, 4, 5), 0, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 1, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 2, [3, 0, 1])
        helper((2, 8, 4, 5), 3, [2, 3, 0])
        helper((2, 3, 3), -1, [1, 2])
        helper((), 0, [0])
        helper((5), 0, [])

    def test_index_select_scalar(self):
        def helper(value, dim, index, idx_dtype=torch.int32):
            cpu_x = torch.tensor(value, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            idx_result = torch.index_select(x, dim=dim, index=idx)
            idx_result_cpu = torch.index_select(cpu_x, dim=dim, index=cpu_idx)

            self.assertEqual(idx_result, idx_result_cpu)

        helper(22, 0, [0])
        with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
            helper(22, 0, [])

    def test_embedding_dense_backward(self):
        def helper(n, d, m, idx):
            embeddingMPS = nn.Embedding(n, d, max_norm=True, device='mps')
            emedding_weight = embeddingMPS.weight.detach().cpu()
            W_MPS = torch.randn((m, d), requires_grad=True, device='mps')
            idx_MPS = torch.tensor(idx, device='mps')
            a_MPS = embeddingMPS.weight.clone() @ W_MPS.t()  # weight must be cloned for this to be differentiable
            a_MPS.retain_grad()
            b_MPS = embeddingMPS(idx_MPS) @ W_MPS.t()  # modifies weight in-place
            b_MPS.retain_grad()
            out_MPS = (a_MPS.unsqueeze(0) + b_MPS)
            loss_MPS = out_MPS.sigmoid().prod()
            loss_MPS.backward()

            embeddingCPU = nn.Embedding(n, d, max_norm=True, _weight=emedding_weight)
            W_CPU = W_MPS.to('cpu')
            idx_CPU = torch.tensor(idx)
            a_CPU = embeddingCPU.weight.clone() @ W_CPU.t()  # weight must be cloned for this to be differentiable
            a_CPU.retain_grad()
            b_CPU = embeddingCPU(idx_CPU) @ W_CPU.t()  # modifies weight in-place
            b_CPU.retain_grad()
            out_CPU = (a_CPU.unsqueeze(0) + b_CPU)
            loss_CPU = out_CPU.sigmoid().prod()
            loss_CPU.backward()

            self.assertEqual(b_CPU.grad, b_MPS.grad)
            self.assertEqual(a_CPU.grad, a_MPS.grad)

        helper(3, 5, 7, [0, 1, 2])
        helper(3, 6, 7, [0, 1, 2])  # verify if changes in shape would cause cached graph lookup problems
        helper(3, 5, 7, 2)  # test scalar index

    # Test pytorch gather
    def test_gather(self):
        def helper(shape, dim, idx_shape, idx_dtype=torch.int64):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = np.random.randint(0, shape[dim], idx_shape)

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            gather_result = torch.gather(x, dim=dim, index=idx)
            gather_result_cpu = torch.gather(cpu_x, dim=dim, index=cpu_idx)

            cpu_grad = torch.randn(idx_shape, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')
            gather_result.backward(gradient=grad)
            gather_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(gather_result, gather_result_cpu)
            self.assertEqual(cpu_x.grad, x.grad)

        helper((6, 3, 3), 0, (3, 3, 3))
        helper((2, 3, 3, 3), 0, (10, 3, 3, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5))
        helper((2, 8, 4, 5), 0, (10, 6, 3, 2))
        helper((8, 8, 4, 5), 0, (6, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (6, 7, 2, 3))
        helper((2, 8, 4, 5), 1, (2, 5, 3, 4))
        helper((2, 8, 4, 5), 2, (1, 8, 10, 3))
        helper((2, 8, 4, 5), 3, (2, 5, 3, 12))

    # Test pytorch gather
    def test_gather_scalar(self):
        idx_dtype = torch.int64
        cpu_x = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
        x = cpu_x.detach().clone().to('mps').requires_grad_()

        idx_np = [0]

        cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
        idx = cpu_idx.detach().clone().to('mps')

        gather_result = torch.gather(x, dim=0, index=idx)
        gather_result_cpu = torch.gather(cpu_x, dim=0, index=cpu_idx)

        cpu_grad = torch.randn([1], device='cpu', dtype=torch.float)
        grad = cpu_grad.to('mps')
        gather_result.backward(gradient=grad)
        gather_result_cpu.backward(gradient=cpu_grad)

        self.assertEqual(gather_result, gather_result_cpu)
        self.assertEqual(cpu_x.grad, x.grad)

    # Test pytorch scatter_add and scatter
    def test_scatter_add(self):
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, do_add=True):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = None
            if (do_add):
                idx_np = np.random.randint(0, shape[dim], idx_shape)
            else:
                idx_np = np.array([[0, 1, 2],
                                   [1, 2, 3],
                                   [2, 3, 4],
                                   [3, 4, 5],
                                   [4, 5, 6]])

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = None
            scatter_result_cpu = None

            if (do_add):
                scatter_result = torch.scatter_add(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)

            cpu_grad = None
            grad = None

            if (idx_shape == src_shape):
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                grad = cpu_grad.to('mps')
                scatter_result.backward(gradient=grad)
                scatter_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(scatter_result, scatter_result_cpu)
            if (idx_shape == src_shape):
                self.assertEqual(cpu_x.grad, x.grad)
                self.assertEqual(cpu_src.grad, src.grad)

        helper((2, 3), 0, (5, 3), (5, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5))

        helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5))
        helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2))
        helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3))
        helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3))

        helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8))
        helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6))
        helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6))

        # Test scatter src
        helper((8, 3), 0, (5, 3), (5, 3), do_add=False)
        helper((10, 3), 0, (5, 3), (5, 8), do_add=False)

    # Test pytorch scatter_add and scatter for scalar input
    def test_scatter_add_scalar(self):
        def helper(idx_dtype=torch.int64, do_add=True):
            cpu_x = torch.tensor(2, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = [0]

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = None
            scatter_result_cpu = None

            if (do_add):
                scatter_result = torch.scatter_add(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=0, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=0, index=cpu_idx, src=cpu_src)

            cpu_grad = None
            grad = None

            cpu_grad = torch.tensor(1.2, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')
            scatter_result.backward(gradient=grad)
            scatter_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(scatter_result, scatter_result_cpu)
            self.assertEqual(cpu_x.grad, x.grad)
            self.assertEqual(cpu_src.grad, src.grad)

        helper()
        helper(do_add=False)

    # Test pytorch scatter_reduce
    def test_scatter_reduce(self):
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, reduce_str="sum"):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = np.random.randint(0, shape[dim], idx_shape)

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = torch.scatter(x, dim=dim, index=idx, src=src, reduce=reduce_str)
            scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src, reduce=reduce_str)

            self.assertEqual(scatter_result, scatter_result_cpu)

        # for reduce in ["sum", "prod", "amax", "amin"]:
        for reduce_type in ["add", "multiply"]:
            helper((2, 3), 0, (5, 3), (5, 3), reduce_str=reduce_type)
            helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5), reduce_str=reduce_type)

            helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5), reduce_str=reduce_type)
            helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3), reduce_str=reduce_type)
            helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3), reduce_str=reduce_type)

            helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8), reduce_str=reduce_type)
            helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6), reduce_str=reduce_type)
            helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6), reduce_str=reduce_type)

    def test_is_nonzero(self):
        self.assertFalse(torch.is_nonzero(torch.tensor([0.]).to('mps')))
        self.assertTrue(torch.is_nonzero(torch.tensor([1.5]).to('mps')))
        self.assertFalse(torch.is_nonzero(torch.tensor([False]).to('mps')))
        self.assertTrue(torch.is_nonzero(torch.tensor([3]).to('mps')))

    # Test triu
    def test_triu(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            triu_result = torch.triu(x, diag)
            triu_result_cpu = torch.triu(cpu_x, diag)

            cpu_grad = torch.randn(triu_result_cpu.shape)
            grad = cpu_grad.to('mps')

            triu_result.backward(gradient=grad)
            triu_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(triu_result, triu_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))
        helper((2, 8, 4, 5), diag=1)
        helper((2, 8, 4, 5), diag=2)
        helper((2, 8, 4, 5), diag=3)
        helper((2, 8, 4, 5), diag=-1)
        helper((2, 8, 4, 5), diag=-2)
        helper((2, 8, 4, 5), diag=-3)

    # Test inverse
    def test_inverse(self):
        def helper(n):
            cpu_input = torch.randn(n, n, device='cpu')
            mps_input = cpu_input.to('mps')

            cpu_result = torch.linalg.inv(cpu_input)
            mps_result = torch.linalg.inv(mps_input)
            self.assertEqual(cpu_result, mps_result)

        helper(2)
        helper(6)
        helper(3)
        helper(8)

    # Test tril
    def test_tril(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            tril_result = torch.tril(x, diag)
            tril_result_cpu = torch.tril(cpu_x, diag)

            cpu_grad = torch.randn(tril_result_cpu.shape)
            grad = cpu_grad.to('mps')

            tril_result.backward(gradient=grad)
            tril_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(tril_result, tril_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))
        helper((2, 8, 4, 5), diag=1)
        helper((2, 8, 4, 5), diag=2)
        helper((2, 8, 4, 5), diag=3)
        helper((2, 8, 4, 5), diag=-1)
        helper((2, 8, 4, 5), diag=-2)
        helper((2, 8, 4, 5), diag=-3)

    # test eye
    def test_eye(self):
        def helper(n, m, dtype):
            cpu_result = None
            result = None

            if (n == m):
                cpu_result = torch.eye(n, dtype=dtype, device='cpu')
                result = torch.eye(n, dtype=dtype, device='mps')
            else:
                cpu_result = torch.eye(n, m, device='cpu')
                result = torch.eye(n, m, device='mps')

            self.assertEqual(result, cpu_result)

        for dtype in [torch.bool, torch.float16, torch.float32, torch.uint8, torch.int16, torch.int32, torch.int64]:
            helper(2, 2, dtype)
            helper(2, 3, dtype)
            helper(0, 2, dtype)
            helper(0, 0, dtype)
            helper(3, 8, dtype)
            helper(8, 3, dtype)

    # Test diag
    def test_diag(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            diag_result = torch.diag(x, diag)
            diag_result_cpu = torch.diag(cpu_x, diag)

            # cpu_grad = torch.randn(diag_result_cpu.shape)
            # grad = cpu_grad.to('mps')

            # diag_result.backward(gradient=grad)
            # diag_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(diag_result, diag_result_cpu)
            # self.assertEqual(x.grad, cpu_x.grad)

        for shape in [(5, 5), (5, 6), (6, 5), (5,), (6,)]:
            for diag in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
                helper(shape, diag=diag)

    # Test linspace
    def test_linspace(self):
        def helper(start, end, steps, dtype=torch.float32):
            cpu_result = torch.tensor(np.linspace(start, end, steps), dtype=dtype)
            result = torch.linspace(start, end, steps, dtype=dtype, device='mps')
            self.assertEqual(cpu_result, result)

        for dtype in [torch.float32, torch.int32, torch.uint8, torch.int64]:
            helper(2, 5, 10, dtype)
            helper(2, 2, 10, dtype)
            helper(5, 2, 10, dtype)
            helper(2, 2, 0, dtype)

    # Test argange
    def test_arange(self):
        self.assertEqual(np.arange(10), torch.arange(10, device='mps'))
        self.assertEqual(np.arange(7, 1, -1), torch.arange(7, 1, -1, device='mps'))
        self.assertEqual(np.arange(1, 2, .3, dtype=np.float32), torch.arange(1, 2, .3, device='mps'))
        self.assertEqual(np.arange(6.3, dtype=np.float32), torch.arange(6.3, device='mps'))

    def test_arange_empty(self):
        out_mps = torch.tensor([], device="mps")
        out_cpu = torch.tensor([], device="cpu")

        y_mps = torch.arange(0, 0, 1, out=out_mps)
        y_cpu = torch.arange(0, 0, 1, out=out_cpu)
        self.assertEqual(y_mps, y_cpu)

    # Test rgange
    def test_range(self):
        self.assertEqual(np.arange(11, dtype=np.float32), torch.range(0, 10, device='mps'))
        self.assertEqual(np.arange(7, 0, -1, dtype=np.float32), torch.range(7, 1, -1, device='mps'))
        self.assertEqual(np.array([1.0000, 1.3000, 1.6000, 1.9000], dtype=np.float32), torch.range(1, 2, .3, device='mps'))
        self.assertEqual(np.arange(6.3, dtype=np.float32), torch.arange(0, 6.3, device='mps'))

    # Test softmax
    def test_softmax(self):
        def helper(shape, dim, channels_last=False):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)

            # Currently NOT testing backward for channels last backward
            cpu_grad = None
            grad = None

            if (not channels_last):
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                grad = cpu_grad.to('mps')

                softmax_result.backward(gradient=grad)
                softmax_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softmax_result, softmax_result_cpu)
            if (not channels_last):
                self.assertEqual(x.grad, cpu_x.grad)

        def helper2(dim):
            cpu_x = torch.tensor(1.23, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)

            cpu_grad = torch.tensor(2.34, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')

            softmax_result.backward(gradient=grad)
            softmax_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softmax_result, softmax_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper2(0)

        for channels_last in [False]:
            for shape in [(2, 4, 8, 5), (3, 4, 6, 7, 2)]:
                if (len(shape) != 4 and channels_last):
                    continue
                for dim in [0, 1, 2, 3, -1, -2, -3]:
                    helper(shape, dim, channels_last)

    def test_nan_to_num(self):
        inputCPU = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
        inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
        outputCPU = torch.nan_to_num(inputCPU, nan=2.0, posinf=1.0, neginf=-1.0)
        outputMPS = torch.nan_to_num(inputMPS, nan=2.0, posinf=1.0, neginf=-1.0)
        self.assertEqual(outputMPS, outputCPU)

    # Test where
    def test_where(self):
        def helper(shape, x_shape, y_shape, cond_dtype=torch.bool, x_dtype=torch.float):

            cpu_cond = torch.randint(2, shape, device='cpu', dtype=cond_dtype, requires_grad=False)
            cond = cpu_cond.detach().clone().to('mps')

            cpu_x = torch.randn(x_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_y = torch.randn(y_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            y = cpu_y.detach().clone().to('mps').requires_grad_()

            cpu_out = torch.where(cpu_cond, cpu_x, cpu_y)
            out = torch.where(cond, x, y)

            cpu_grad = torch.randn(cpu_out.shape)
            grad = cpu_grad.to('mps')

            cpu_out.backward(gradient=cpu_grad)
            out.backward(gradient=grad)

            self.assertEqual(out, cpu_out)
            self.assertEqual(x.grad, cpu_x.grad)
            self.assertEqual(y.grad, cpu_y.grad)

        for shape in ([(0, 3), [], (2, 3), (9,)]):
            helper(shape, shape, shape)

        helper((2, 3, 1), (2, 3, 4), (2, 1, 4))
        helper((2, 1, 1), (2, 3, 4), (1, 3, 4))
        helper((1, 1, 1), (1, 1, 4), (2, 3, 1))
        helper([], (1, 1, 4), (2, 3, 1))
        helper([], (2, 3, 4), [])
        helper((5, 2, 3), (2, 3), (2, 3))
        helper((2, 3), (5, 2, 3), (2, 3))
        helper((2, 3), (2, 3), (5, 2, 3))
        helper((2, 3), (5, 2, 3), (6, 5, 2, 3))

    # Test normal
    def test_normal(self):
        def helper(shape, mean=0.0, std=1.0):
            mps_out = torch.normal(mean, std, shape, device='mps')

            mean_array = np.ones(shape)
            mean_array *= mean
            cpu_mean_tensor = torch.tensor(mean_array, device='cpu', dtype=torch.float, requires_grad=False)
            mean_tensor = cpu_mean_tensor.detach().clone().to('mps')

            std_array = np.ones(shape)
            std_array *= std
            cpu_std_tensor = torch.tensor(std_array, device='cpu', dtype=torch.float, requires_grad=False)
            std_tensor = cpu_std_tensor.detach().clone().to('mps')

            # test out
            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean, std_tensor, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std_tensor, out=mps_out)

            # test without out
            mps_out = torch.normal(mean_tensor, std)
            self.assertEqual(mps_out.size(), mean_tensor.size())

            mps_out = torch.normal(mean, std_tensor)
            self.assertEqual(mps_out.size(), std_tensor.size())

            inferred_shape = torch.broadcast_shapes(mean_tensor.size(), std_tensor.size())
            mps_out = torch.normal(mean_tensor, std_tensor)
            self.assertEqual(mps_out.size(), inferred_shape)

        helper((2, 3, 4, 5, 6))
        helper((100, 100), 2.5, 1.2)

    def test_bernoulli(self):
        shape = (10, 10)
        all_ones = torch.ones(shape, device='mps')
        all_zeros = torch.zeros(shape, device='mps')

        prob_tensor = all_ones * 0.5
        # probability of drawing "1" is 0.5
        mps_out = torch.bernoulli(prob_tensor)
        # We can't check reliably the mean and std.
        # Just make sure we don't return constant values
        self.assertNotEqual(mps_out.to('cpu').mean(), 0.)
        self.assertNotEqual(mps_out.to('cpu').std() ** 2, 0.)

        # probability of drawing "1" is 0
        mps_out = torch.bernoulli(all_zeros)
        self.assertEqual(mps_out, all_zeros)

        # probability of drawing "1" is 1
        mps_out = torch.bernoulli(all_ones)
        self.assertEqual(mps_out, all_ones)

        # Check it works for different dtypes
        for dtype in [torch.float16, torch.int8, torch.int16, torch.int32, torch.int64]:
            mps_out = torch.zeros(shape, device='mps', dtype=dtype).bernoulli(0.5)
            # Check that output is not all zeros or ones
            if product_version > 13.0:
                uniq = mps_out.unique()
                self.assertEqual(uniq, torch.arange(2, device='mps', dtype=dtype))
            else:
                self.assertEqual(mps_out.min().item(), 0.)
                self.assertEqual(mps_out.max().item(), 1.)

    def test_mps_generator(self):
        # explicit manual seeding by creating an MPS Generator
        g_mps = torch.Generator(device='mps')
        g_mps.manual_seed(999)
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        g_mps.manual_seed(999)
        mps_y = torch.randn(5, device='mps', generator=g_mps)
        # seed values were the same, so the random tensor contents should match
        self.assertEqual(mps_x, mps_y)
        # save generator's state to restore it later
        g_state = g_mps.get_state()

        # generate random numbers without seeding
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        # in this case, the random results must differ from the last generated random results
        self.assertNotEqual(mps_x, mps_y)

        # restore the previously saved state, and the results should match again
        g_mps.set_state(g_state)
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        self.assertEqual(mps_x, mps_y)

    def test_default_mps_generator(self):
        # manual seeding on the "default" MPS generator using
        # the global torch.manual_seed()
        torch.manual_seed(230)
        mps_x = torch.randn(5, device='mps')
        # manual seeding using torch.mps.manual_seed()
        # which should set the "default" MPS generator
        # like the global torch.manual_seed()
        torch.mps.manual_seed(230)
        mps_y = torch.randn(5, device='mps')
        # seed values were the same, so the random tensor contents should match
        self.assertEqual(mps_x, mps_y)

        # save the default generator's state to restore it later
        g_state = torch.mps.get_rng_state()

        # generate random numbers without seeding
        mps_x = torch.randn(5, device='mps')
        # in this case, the random results must differ from the last generated random results
        self.assertNotEqual(mps_x, mps_y)

        # restore the previously saved state, and the results should match again
        torch.mps.set_rng_state(g_state)
        mps_x = torch.randn(5, device='mps')
        self.assertEqual(mps_x, mps_y)

    def test_device_synchronize(self):
        # just running some ops each followed by a synchronize to wait for
        # MPS stream to finish running each of them
        net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
            .to(device='mps', dtype=torch.float)

        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        torch.mps.synchronize()
        x = net1(x)
        torch.mps.synchronize()
        x.backward(torch.randn_like(x))
        torch.mps.synchronize()

    @unittest.expectedFailure
    def test_mps_allocator_module(self):
        # first garbage collect and empty the cached blocks
        gc.collect()
        torch.mps.empty_cache()
        # measure memory allocations from MPSAllocator
        current_alloc_before = torch.mps.current_allocated_memory()
        # after garbage collection and emptying the cache the
        # current_allocated_memory must be zero
        self.assertTrue(current_alloc_before == 0)
        # measure total memory allocations from Metal driver
        driver_alloc_before = torch.mps.driver_allocated_memory()
        # allocate a new 8 MB tensor to force allocation of a new Metal Heap
        x = torch.ones(1024 * 1024 * 8, device="mps")
        # get memory allocations after allocating tensor x
        current_alloc_after = torch.mps.current_allocated_memory()
        driver_alloc_after = torch.mps.driver_allocated_memory()
        # current and driver memory allocations must have
        # grown at this point
        self.assertTrue(current_alloc_after > current_alloc_before)
        self.assertTrue(driver_alloc_after > driver_alloc_before)

    # to verify this test, run XCode Instruments "Metal System Trace" or "Logging" tool,
    # press record, then run this python test, and press stop. Next expand
    # the os_signposts->PyTorchMPS and check if events or intervals are logged
    # like this example:
    # "aten::mps_convolution_backward_input:f32[1,128,6,6]:f32[128,64,3,3]:1,128,6,6 (id=G2, run=2)"
    def test_mps_profiler_module(self):
        with torch.mps.profiler.profile(mode="event", wait_until_completed=False) as p:
            # just running some ops to capture the OS Signposts traces for profiling
            net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
                .to(device='mps', dtype=torch.float)
            x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
            x = net1(x)

        torch.mps.profiler.start(mode="interval", wait_until_completed=True)
        # just running some ops to capture the OS Signposts traces for profiling
        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        x = net1(x)
        torch.mps.profiler.stop()

    def test_mps_event_module(self):
        startEvent = torch.mps.Event(enable_timing=True)
        startEvent.record()
        net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
            .to(device='mps', dtype=torch.float)
        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        x = net1(x)
        endEvent = torch.mps.Event(enable_timing=True)
        endEvent.record()
        elapsedTime = startEvent.elapsed_time(endEvent)
        self.assertTrue(elapsedTime > 0.0)

    def test_jit_save_load(self):
        m = torch.nn.Module()
        m.x = torch.rand(3, 3, device='mps')
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        n = torch.jit.load(buffer)
        self.assertEqual(n.x, m.x)

    # Test random_, random_.to and random_.from
    def test_random(self):
        def helper(shape, low, high, dtype=torch.int32):

            mps_out = torch.randint(low, high, shape, dtype=dtype, device='mps')

            # We can't check reliably the mean and std.
            # Just make sure we don't return constant values
            self.assertNotEqual(mps_out.float().mean().item(), 0.)
            self.assertNotEqual(mps_out.float().std().item(), 0.)

        helper([100, 100], 0, 10)
        helper([100, 100], 23, 89)
        helper([100, 100], 23, 89, dtype=torch.float32)
        helper([100, 100], 23, 89, dtype=torch.int64)
        helper([100, 100], 0, 2, dtype=torch.bool)

        # Test random_
        for dtype in [torch.bool, torch.int8, torch.uint8, torch.int32, torch.float16, torch.float32]:
            x = torch.empty(10, 10, dtype=dtype, device='mps')
            x.random_()
            self.assertNotEqual(x.max().item(), 0)

    # Test exponential
    def test_exponential(self):
        def helper(shape, lamda, dtype=torch.float32):

            mps_out = torch.zeros(shape, device='mps', dtype=dtype)
            mps_out.exponential_(lamda)

            print(mps_out.to('cpu').float().mean(), 1 / lamda)
            print(mps_out.to('cpu').float().std() ** 2, 1 / (lamda**2))

        for dtype in [torch.float32, torch.float16]:
            helper([100, 100], 2, dtype)
            helper([100, 100], 1, dtype)
            helper([100, 100], 3, dtype)
            helper([100, 100], 0.5, dtype)

    def test_exponential_1(self):
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

    # Test add
    def test_add_sub(self):
        def helper(shape, alpha, op_name, inplace):
            if op_name == "add":
                op = torch.Tensor.add_ if inplace else torch.add
            elif op_name == "sub":
                op = torch.Tensor.sub_ if inplace else torch.sub

            for dtype in [torch.float16, torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')

                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_y = cpu_y.detach().clone().to('mps')

                cpu_out = op(cpu_x, cpu_y, alpha=alpha)
                mps_out = op(mps_x, mps_y, alpha=alpha)
                # fp16 isn't accurate when alpha is passed
                # TODO: remove or fix 'tol' when we fix problems with fp16
                tol = 2e-3 if dtype is torch.float16 else None
                self.assertEqual(mps_out, cpu_out, rtol=tol, atol=tol)
                if not (cpu_y.shape != () and inplace):  # in-place output cannot be broadcasted.
                    # create a scalar tensor
                    cpu_s = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                    mps_s = cpu_s.detach().clone().to('mps')
                    # primary tensor is scalar
                    self.assertEqual(op(cpu_s, cpu_y), op(mps_s, mps_y))
                # create a scalar tensor
                cpu_s = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                mps_s = cpu_s.detach().clone().to('mps')
                # secondary tensor is scalar
                self.assertEqual(op(cpu_x, cpu_s), op(mps_x, mps_s), rtol=tol, atol=tol)


        for op_name, inplace in product(["add", "sub"], [True, False]):
            helper((), 0.0, op_name, inplace)
            helper((2, 8, 4, 5), 0.0, op_name, inplace)
            helper((2, 8, 4, 5), 0.1, op_name, inplace)
            helper((2, 8, 4, 5), 1.0, op_name, inplace)
            helper((2, 8, 3, 5), 0.1, op_name, inplace)
            helper((2, 8, 3, 5), 0.2, op_name, inplace)

    # Test add
    def test_add_scalars(self):
        def helper(alpha):
            for dtype in [torch.float16, torch.float32]:
                cpu_x = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')

                cpu_y = torch.tensor(3.4, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')

                cpu_out = torch.add(cpu_x, cpu_y, alpha=alpha)
                out = torch.add(x, y, alpha=alpha)
                # fp16 isn't accurate when alpha is passed
                tol = 1e-3 if dtype is torch.float16 else None
                self.assertEqual(out, cpu_out, rtol=tol, atol=tol)

        helper(1.0)
        helper(0.0)
        helper(0.1)
        helper(0.2)

        # Test int32 tensor + int64 scalar add
        # see https://github.com/pytorch/pytorch/issues/79835#issuecomment-1164984534
        x = torch.ones(4, dtype=torch.int32, device='mps')
        self.assertEqual(x + 1, torch.full((4,), 2, dtype=torch.int32, device='mps'))
        self.assertTrue(torch.equal(x + 1.5, torch.full((4,), 2.5, device='mps')))

    def test_types_binary_op(self):
        # Float * Bool
        cpu_x = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([True, False, True, False, True], device="cpu")
        mps_x = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([True, False, True, False, True], device="mps")
        self.assertEqual(cpu_x, mps_x)
        # Float * Int64
        cpu_y = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([1, 0, 1, 0, 1], device="cpu")
        mps_y = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([1, 0, 1, 0, 1], device="mps")
        self.assertEqual(cpu_y, mps_y)

    def test_unary_ops(self):
        def helper(shape, op):
            for dtypef in [torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtypef, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                self.assertEqual(op(cpu_x), op(mps_x))

            for dtypei in [torch.int32, torch.int16]:
                cpu_x = torch.randint(0, 1000, shape, device='cpu', dtype=dtypei, requires_grad=False)
                mps_x = cpu_x.to('mps')
                self.assertEqual(op(cpu_x), op(mps_x), rtol=1e-4, atol=1e-4)

        helper((2, 8, 4, 5), torch.exp)
        helper((2, 8, 3, 5), torch.exp2)
        helper((2, 8, 3, 5), torch.expm1)
        helper((2, 8, 3, 5), torch.log)
        helper((2, 8, 3, 5), torch.cos)
        helper((2, 8, 3, 5), torch.erfinv)


    def test_non_dense_in_storage_unary_ops(self):
        def helper(op):
            for dtypef in [torch.float32]:
                cpu_x = torch.randn(100, device='cpu', dtype=dtypef, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                self.assertEqual(op(cpu_x[::2]), op(mps_x[::2]))

            for dtypei in [torch.int32, torch.int16, torch.int8]:
                cpu_x = torch.randint(127, device='cpu', size=(100,), dtype=dtypei, requires_grad=False)
                mps_x = cpu_x.to('mps')
                self.assertEqual(op(cpu_x[::2]), op(mps_x[::2]), rtol=1e-4, atol=1e-4)

        helper(torch.exp)
        helper(torch.exp2)
        helper(torch.expm1)
        helper(torch.log)
        helper(torch.cos)

    def test_atan2(self):
        def helper(shape):
            input_cpu = torch.randn(shape)
            input_mps = input_cpu.detach().clone().to("mps")

            other_cpu = torch.randn(shape)
            other_mps = other_cpu.detach().clone().to("mps")

            atan2_cpu = torch.atan2(input_cpu, other_cpu)
            atan2_mps = torch.atan2(input_mps, other_mps)

            self.assertEqual(atan2_cpu, atan2_mps.to("cpu"))

        helper(4)
        helper(10000)
        helper((10000, 40))

    def test_multinomial(self):
        # Test with num_dist = 1
        def helper(probs, compare_mean, compare_var, num_samples=5, replacement=True):
            cpu_prob_tensor = torch.tensor(probs, device='cpu', dtype=torch.float, requires_grad=False)
            prob_tensor = cpu_prob_tensor.detach().clone().to('mps')

            mps_out = torch.multinomial(prob_tensor, num_samples, replacement=replacement)
            if (not replacement):
                print(mps_out.to('cpu'))
            else:
                # Compare "real" with theoretical values
                print(mps_out.to('cpu').float().mean(), compare_mean)
                print(mps_out.to('cpu').float().std() ** 2, compare_var)

        # TODO: Add tests for data types
        helper(np.array([[0., 0., 0., 0.5, 0.5]]), (3 + 4) / 2, (12.5 - 3.5 ** 2), 100000)
        helper(np.array([[.2, .2, .2, .2, .2]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([1, 1, 1, 1, 1]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1, 1, 1]]), 0, 0, 7, False)

    def test_cumsum_dim_check(self):
        x = torch.rand((3, 3), device="mps")
        self.assertEqual(x.cumsum(1), x.cumsum(-1))
        self.assertEqual(x.cumsum(0), x.cumsum(-2))
        self.assertRaises(IndexError, lambda: x.cumsum(2))
        self.assertRaises(IndexError, lambda: x.cumsum(-3))

    def test_cumprod_dim_check(self):
        x = torch.rand((3, 3), device="mps")
        self.assertEqual(x.cumprod(1), x.cumprod(-1))
        self.assertEqual(x.cumprod(0), x.cumprod(-2))
        self.assertRaises(IndexError, lambda: x.cumprod(2))
        self.assertRaises(IndexError, lambda: x.cumprod(-3))


class TestTopK(TestCase):
    def _test_topk(self, shape, largest):
        cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
        x = cpu_x.detach().clone().to('mps')
        if isinstance(shape, tuple):
            for curr_dim, dim_size in enumerate(shape):
                for k in range(1, dim_size + 1):
                    topk_values, topk_indices = torch.topk(x, k, dim=curr_dim, largest=largest)
                    topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=curr_dim, largest=largest)
                    self.assertEqual(topk_values, topk_values_cpu)
                    self.assertEqual(topk_indices, topk_indices_cpu)
        else:
            for k in range(1, shape):
                topk_values, topk_indices = torch.topk(x, k, dim=0, largest=largest)
                topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=0, largest=largest)
                self.assertEqual(topk_values, topk_values_cpu)
                self.assertEqual(topk_indices, topk_indices_cpu)

    def test_topk(self):
        largest_vals = [True, False]
        shapes = [
            # Zero Element Tensors
            0,
            (1, 0),
            (0, 1),
            (1, 0, 1),
            # Multiple Element Tensors
            1,
            2,
            (5, 1),
            (1, 5),
            (5, 9, 7, 4),
        ]

        for shape in shapes:
            for largest_val in largest_vals:
                with self.subTest(shape=shape, largest_val=largest_val):
                    self._test_topk(shape, largest_val)

class TestNNMPS(NNTestCase):

    def _create_basic_net(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    def test_requires_grad_(self):
        m = self._create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    def test_module_backcompat(self):
        from torch.serialization import SourceChangeWarning
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path)
        input = torch.randn(2, 3, dtype=torch.float)
        self.assertEqual(m(input).size(), (2, 5))

    def test_conv_backcompat(self):
        from torch.serialization import SourceChangeWarning
        # This file was generated by running on PyTorch 1.0.1 on Python 2:
        #
        #     import torch
        #     from torch import nn
        #     m = nn.Conv2d(1, 1, 1)
        #     torch.save(m, 'legacy_conv2d.pt')
        #
        # NB: This Pickle also contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path, encoding='utf-8')
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        self.assertEqual(m(input).size(), (1, 1, 1, 1))

    def test_conv_expand(self):
        device = 'mps'
        input_ = torch.rand(2, 3, 16, 16, device=device)
        kernel = torch.rand(1, 1, 3, 11, device=device)
        tmp_kernel = kernel.expand(-1, 3, -1, -1)
        output = F.conv2d(input_, tmp_kernel, groups=1, padding=0, stride=1)

    # The test should not crash
    def test_permute(self):
        M_cpu = torch.randn(5, 5)
        M_mps = M_cpu.to('mps')

        output_cpu = M_cpu.permute(1, 0)
        output_mps = M_mps.permute(1, 0)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    # Printing of non_contiguous should not crash
    def test_print_non_contiguous(self):
        print(torch.ones(100, 100, device='mps').nonzero())
        print(torch.ones(100, 100, device='mps').nonzero().contiguous())

    def test_zero_grad(self):
        i = torch.randn(2, 5, requires_grad=True)
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertIsNone(module.weight.grad)

        module.bias.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)

        # Force set to zeros.
        module.zero_grad(set_to_none=False)
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)


    def test_no_grad(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            output = module(x)
            self.assertTrue(output.requires_grad)
            output.backward(torch.ones(1, 5, 10, 10))

            with torch.no_grad():
                output2 = module(y)
                self.assertFalse(output2.requires_grad)
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))

    def test_invalid_conv1d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv1d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(4\). ' +
                                        r'Kernel size: \(10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_conv2d_discontiguous_weight(self):
        # Test for https://github.com/pytorch/pytorch/issues/55781
        x = torch.ones(64, 16, 16, 16)
        weight = torch.arange(0, 1.0, 1 / 2.0 ** 10).reshape(32, 16, 1, 2)[:, :, :, ::2]
        self.assertFalse(weight.is_contiguous())
        y = torch.nn.functional.conv2d(x, weight, None)
        if torch.backends.mkldnn.is_available():
            # Disable MKLDNN explicitly, so that either NNPACK or THCNN will be used
            with torch.backends.mkldnn.flags(enabled=False):
                y_ = torch.nn.functional.conv2d(x, weight, None)
                self.assertEqual(y, y_)
        self.assertEqual(y.sum(), 4186112.)

    def test_invalid_conv2d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True)
            input = torch.randn(1, 3, 1, 1)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Zero stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Input and weights on different devices
            self.assertRaisesRegex(RuntimeError,
                                   'must be on the same device',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 3, 3, device='mps')))
            self.assertRaisesRegex(RuntimeError,
                                   'Input type \\(MPSFloatType\\) and weight type \\(torch\\.FloatTensor\\) should be the same',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32, device='mps'), torch.rand(1, 3, 3, 3)))


    def test_conv2d_valid_padding(self, device='mps'):
        # Test F.conv2d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 10, device=device).to(torch.float)
        y = torch.rand(1, 1, 1, 4, device=device).to(torch.float)

        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding='valid')
        self.assertEqual(expect.to('cpu'), actual.to('cpu'))

    def test_gemm_permute_transpose(self):
        batch_size = 32
        n = 20
        hidden = 768
        num_attention_heads = 12
        attention_head_size = hidden // num_attention_heads

        def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def attention2(key, *, workaround=False, device):
            key = transpose_for_scores(key)
            res = key.transpose(-1, -2)
            return res

        A = torch.randn(batch_size, n, hidden)
        A_mps = A.detach().clone().to("mps")

        r1 = attention2(A, device="cpu")
        r2 = attention2(A_mps, device="mps")

        r2_cpu = r2.to("cpu")
        self.assertEqual(r1, r2_cpu)

    def test_group_norm_backward(self, device='mps'):
        # See https://github.com/pytorch/pytorch/issues/88331 for more detail
        shape = [1, 4, 16, 16]
        x = torch.full(shape, 7.0, device=device)

        target = torch.ones((1, 3, 128, 128), device=device)

        conv_in = nn.Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
        conv_out = nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
        norm = nn.GroupNorm(32, 128, eps=1e-6, affine=True, device=device)

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            out = 5.5 * x
            out = conv_in(out)
            out = out + norm(out)
            out = out + norm(out)
            out = out + norm(out)
            out = F.interpolate(out, scale_factor=8.0, mode="nearest")
            out = norm(out)
            out = conv_out(out)

            loss = (out - target).norm(dim=-1).sum()
            grad = -torch.autograd.grad(loss, x)[0]
            self.assertFalse(grad.detach().isnan().any().item(), 'NaN gradients returned by autograd')


    # def test_conv2d_same_padding(self, device='mps'):
        # x = torch.rand(1, 1, 10, 11, device=device)
        # y = torch.rand(1, 1, 4, 5, device=device)
        # expect = F.conv2d(x, y, padding=(2, 2))[..., 1:, :]
        # actual = F.conv2d(x, y, padding='same')
        # self.assertEqual(expect.to('cpu'), actual.to('cpu'))

        # # With dilation
        # y = torch.rand(1, 1, 3, 4, device=device)
        # expect = F.conv2d(x, y, padding=(2, 3), dilation=2)
        # actual = F.conv2d(x, y, padding='same', dilation=2)
        # self.assertEqual(expect, actual)

        # # Dilation with asymmetric padding
        # y = torch.rand(1, 1, 4, 4, device=device)
        # expect = F.conv2d(x, y, padding=5, dilation=3)[..., 1:, 1:]
        # actual = F.conv2d(x, y, padding='same', dilation=3)
        # self.assertEqual(expect, actual)


class TestConstantPadNd(TestCaseMPS):
    def test_preserves_memory_format(self):
        nchw_tensor = torch.rand((1, 2, 5, 3))
        nchw_padded = torch.constant_pad_nd(nchw_tensor, [1, 2], 0.5)
        self.assertTrue(nchw_padded.is_contiguous(memory_format=torch.contiguous_format))

        nhwc_tensor = nchw_tensor.contiguous(memory_format=torch.channels_last)
        nhwc_padded = torch.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)
        self.assertTrue(nhwc_padded.is_contiguous(memory_format=torch.channels_last))


class TestLinalgMPS(TestCaseMPS):
    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False):
        dtype = t.dtype
        numpy_dtype = dtype
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
        if beta != 0:
            res3 += (torch.mul(t, beta)).to(numpy_dtype).cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    def test_addmm(self, device="mps", dtype=torch.float32):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2, beta=0)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

        M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
        m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
        m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
        self._test_addmm_addmv(torch.addmm, M, m1, m2, transpose_out=t4)

    def _test_addr(self, f, t, m, v, alpha=None, beta=None):
        dtype = t.dtype
        numpy_dtype = dtype
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = alpha * np.outer(m.to(numpy_dtype).cpu().numpy(), v.to(numpy_dtype).cpu().numpy())
        if beta != 0:
            res2 += (torch.mul(t, beta)).to(numpy_dtype).cpu().numpy()
        res2 = torch.from_numpy(res2).to(dtype)
        self.assertEqual(res1, res2)

    def test_addr(self, device="mps", dtype=torch.float32):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, device=device).to(dtype)
        m2 = torch.randn(25, device=device).to(dtype)
        self._test_addr(torch.addr, M, m1, m2)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, device=device).to(dtype)
        m2 = torch.randn(25, device=device).to(dtype)
        self._test_addr(torch.addr, M, m1, m2, beta=0)

class TestGatherScatter(TestCaseMPS):
    def test_slicing_with_step(self):
        # Slicing with step
        # https://github.com/pytorch/pytorch/issues/78886
        x_mps = torch.zeros(10, dtype=torch.float32, device="mps")
        x_mps[::2] = 1.0

        x_cpu = torch.zeros(10, dtype=torch.float32, device="cpu")
        x_cpu[::2] = 1.0

        self.assertEqual(x_cpu, x_mps)

    def test_cast_gather_scatter(self):
        for _ in range(0, 50):
            input = np.random.randint(0, 255, size=(5, 5, 4), dtype=np.uint8)
            with torch.no_grad():
                s = torch.tensor(input, dtype=torch.uint8, device="mps").unsqueeze(0)
                s_cpu = torch.tensor(input, dtype=torch.uint8, device="cpu").unsqueeze(0)
                s = s.long()
                s_cpu = s_cpu.long()
                self.assertEqual(s.cpu(), s_cpu)

                s = s.float()
                s_cpu = s_cpu.float()
                self.assertEqual(s.cpu(), s_cpu)

                s /= 255
                s_cpu /= 255
                self.assertEqual(s.cpu(), s_cpu)

    def test_slicing_replace_column(self):
        # https://github.com/pytorch/pytorch/issues/78074
        def _helper(tensor_data):
            x_cpu = torch.tensor(tensor_data)
            x_mps = x_cpu.to('mps')

            x_cpu[:, 0] = 7
            x_mps[:, 0] = 7

            self.assertEqual(x_cpu, x_mps)

        _helper([[1, 2, 3], [4, 5, 6]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    def test_inplace_scatter(self):
        # https://github.com/pytorch/pytorch/issues/79672
        a_mps = torch.ones((2, 2),).to(torch.device("mps"))
        b_mps = torch.ones((2, 2),).to(torch.device("mps"))

        a_cpu = torch.ones((2, 2),).to(torch.device("cpu"))
        b_cpu = torch.ones((2, 2),).to(torch.device("cpu"))

        a_mps[:, 0] += b_mps[:, 0]
        a_cpu[:, 0] += b_cpu[:, 0]
        self.assertEqual(a_cpu, a_mps)

        a_mps[:, 0] = a_mps[:, 0] + b_mps[:, 0]
        a_cpu[:, 0] = a_cpu[:, 0] + b_cpu[:, 0]
        self.assertEqual(a_cpu, a_mps)

# These tests were taken from test/test_view_ops.py
# They are subset of those tests as currently only this subset is working.
# This whole `class` will be removed when we add generic device testing. There
# are no additional tests added apart from what is part of test_view_ops.py
class TestViewOpsMPS(TestCaseMPS):
    exact_dtype = True

    def test_permute_slicing(self):
        # test the fix for crash reported in
        # https://github.com/pytorch/pytorch/issues/94190
        cpu_x = (torch.randn([3, 2, 2]).float())
        mps_x = cpu_x.detach().clone().to('mps')
        cpu_out = cpu_x.permute((2, 0, 1)) * 2.0
        mps_out = mps_x.permute((2, 0, 1)) * 2.0
        # this print caused a crash prior to fix PR#94259
        print(torch.zeros_like(mps_out))
        # test the fix for fill_scalar_mps() mentioned in issue #94190
        self.assertEqual(torch.zeros_like(cpu_out), torch.zeros_like(mps_out))
        self.assertEqual(cpu_x[:, 1, :].fill_(1), mps_x[:, 1, :].fill_(1))

    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False
        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == 'mps':
            if base.storage().data_ptr() != other.storage().data_ptr():
                return False

        return True

    # Returns true if v1 and v2 are views of the same base
    def is_view_of_same_base(self, v1, v2):
        if (not v1._is_view() or v1 is v2):
            return False
        return self.is_view_of(v1._base, v2)

    # Performs transpose if contiguous=True, else returns the input tensor as is
    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    def test_diagonal_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = torch.diagonal(t)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        t = torch.ones((3, 3, 3), device="mps")
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    def test_select_view(self, device="mps") -> None:
        t = torch.ones((5, 5), device=device)
        v = t.select(0, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    def test_unbind_view(self, device="mps") -> None:
        t = torch.zeros((5, 5), device=device)
        tup = torch.unbind(t)

        for idx, v in enumerate(tup):
            self.assertTrue(self.is_view_of(t, v))

            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    def test_expand_view(self, device="mps") -> None:
        t = torch.ones((5, 1), device=device)
        v = t.expand(5, 5)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_expand_as_view(self, device="mps"):
        t = torch.ones((5, 1), device=device)
        e = torch.empty((5, 5), device=device)
        v = t.expand_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_narrow_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = torch.narrow(t, 1, 2, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 2], v[0, 0])

    def test_permute_view(self, device="mps") -> None:
        t = torch.ones((5, 5), device=device)
        v = t.permute(1, 0)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_view(self, device="mps"):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            t = torch.ones((5, 5), device=device)
            v = fn(t, 0, 1)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapdims_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapaxes_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.transpose_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = t.t()
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.t_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_T_view(self, device="mps"):
        for op in ("T", "H", "mT", "mH"):
            t = torch.ones((5, 5), device=device)
            v = getattr(t, op)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_unfold_view(self, device="mps"):
        t = torch.ones(10, device=device)
        v = t.unfold(0, 3, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[1, 0] = 0
        self.assertEqual(t[2], v[1, 0])

    def test_squeeze_view(self, device="mps"):
        t = torch.ones(5, 1, 5, device=device)
        v = torch.squeeze(t)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertTrue(t is v._base)

    def test_squeeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.squeeze_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertTrue(t is v._base)

    def test_unsqueeze_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.unsqueeze(t, 1)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_unsqueeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.unsqueeze_(1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_as_strided_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.as_strided(t, (25,), (1,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_as_strided_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.as_strided_((25,), (1,))
        self.assertTrue(self.is_view_of(t, v))
        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view(25)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,))
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_contiguous_self(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        s = t.contiguous()
        self.assertTrue(s is t)

    def test_contiguous_nonview(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])

    def test_reshape_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.reshape(t, (25,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_as_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,), device=device)
        v = t.reshape_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_nonview(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        nv = torch.reshape(t.t(), (25,))
        self.assertTrue(not self.is_view_of(t, nv))

        nv[6] = 0
        self.assertNotEqual(t[1, 1], nv[6])

    def test_flatten_view(self, device="mps"):
        def test_writes_propagate(t, v):
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            v[idx_v] = 0
            self.assertEqual(t[idx_t], v[idx_v])

        t = torch.ones(1, 2, 3, 4, device=device)
        v = t.flatten()
        self.assertTrue(self.is_view_of(t, v))
        test_writes_propagate(t, v)

        # zero-dimensional tensor
        t = torch.tensor(1, device=device)
        v = t.flatten()
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of(t, v))

        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        v = t.flatten(0, 1)
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of_same_base(t, v))

        # stride[i] = stride[i + 1] * size[i + 1] is satisfied for 3 groups:
        t = torch.ones(720, device=device) \
            .as_strided((2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0))
        #               [--1--|---2---|-3-] [--1--|----2---|-3-]
        v1 = t.flatten(0, 1)
        v2 = v1.flatten(1, 3)
        v3 = v2.flatten(2, 2)
        test_writes_propagate(t, v1)
        self.assertTrue(self.is_view_of_same_base(t, v1))
        test_writes_propagate(t, v2)
        self.assertTrue(self.is_view_of_same_base(t, v2))
        test_writes_propagate(t, v3)
        self.assertTrue(self.is_view_of_same_base(t, v3))

    def test_flatten_nonview(self, device="mps"):
        def assert_is_nonview(t, nv):
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            self.assertTrue(not nv._is_view())
            nv[idx_nv] = 0
            self.assertNotEqual(t[idx_t], nv[idx_nv])
        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        nv = t.flatten(1, 3)
        assert_is_nonview(t, nv)

        t = torch.ones(2, 2, device=device).T
        nv = t.flatten()
        assert_is_nonview(t, nv)

        # flatten returns the original object if start_dim=end_dim
        t = t = torch.ones(2, 2, device=device)
        nv = t.flatten(1, 1)
        self.assertTrue(t is nv)

    def test_basic_indexing_slice_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[:2, :3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_ellipses_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[..., :2]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_newaxis_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[None, :2, 3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 3], v[0, 0])

    def test_chunk_view(self, device="mps"):
        t = torch.zeros(3, 3, device=device)
        l = torch.chunk(t, 3)

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_split_view(self, device="mps"):
        t = torch.zeros(3, 3, device=device)
        l = torch.split(t, [1, 1, 1])

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_movedim_view(self, device="mps"):
        def run_test(device, op):
            t = torch.zeros(3, 3, device=device)
            out = op(t)

            self.assertTrue(self.is_view_of(t, out))

            # Randomly change values in output
            # and verify that original is changed
            # as well.
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        for fn in [torch.movedim, torch.moveaxis]:
            op = partial(fn, source=(0, 1), destination=(1, 0))
            run_test(device, op)

            op = partial(fn, source=0, destination=1)
            run_test(device, op)

    # Testing that the generated view_copy kernel and its derivative are implemented correctly
    def test_view_copy(self, device="mps"):
        a = torch.randn(4, device=device, requires_grad=True)
        a_ref = a.clone().detach().requires_grad_()
        a_view = a_ref.view(2, 2)
        a_view_copy = torch.view_copy(a, (2, 2))

        # view_copy ops don't preserve view relationship
        self.assertTrue(self.is_view_of(a_ref, a_view))
        self.assertFalse(self.is_view_of(a, a_view_copy))

        a_view_copy.sum().backward()
        a_view.sum().backward()

        # forward and backward give the same shape + result
        self.assertEqual(a_view_copy, a_view)
        self.assertEqual(a.grad, a_ref.grad)

    def test_view_copy_out(self, device="mps"):
        a = torch.randn(2, 2, device=device)
        out = torch.empty(2, device=device)

        torch.diagonal_copy(a, out=out)
        expected = torch.diagonal_copy(a)

        self.assertEqual(expected, out)

        a = torch.randn(4, device=device)
        out1 = torch.empty(2, device=device)
        out2 = torch.empty(2, device=device)

        torch.split_copy(a, 2, out=(out1, out2))
        expected1, expected2 = torch.split_copy(a, 2)

        self.assertEqual(expected1, out1)
        self.assertEqual(expected2, out2)

    def test_detached_view_copy(self, device="mps"):
        # https://github.com/pytorch/pytorch/issues/86052
        x = torch.arange(2)
        # .detach() makes y not a view, but contig tensor
        # with non-zero offset
        y = x[1].detach()
        z = y.to(device)
        self.assertEqual(y, z.cpu())

    def test_empty_reshape(self, device="mps"):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # should be viewable -- i.e. data_ptr is the same.
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # match NumPy semantics -- don't infer the size of dimension with a degree of freedom
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    def test_expand(self, device="mps"):
        tensor = torch.rand(1, 8, 1, device=device)
        tensor2 = torch.rand(5, device=device)
        template = torch.rand(4, 8, 5, device=device)
        target = template.size()
        self.assertEqual(tensor.expand_as(template).size(), target)
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor.expand(target).size(), target)
        self.assertEqual(tensor2.expand_as(template).size(), target)
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor2.expand(target).size(), target)

        # test double expand
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # test non-contiguous
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(noncontig.is_contiguous())
        self.assertEqual(noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1))

        # make sure it's compatible with unsqueeze
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # test -1 as target size
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # test expanding empty to empty
        self.assertEqual(torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device))

    def test_view_empty(self, device="mps"):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    def test_reshape(self, device="mps"):
        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        # .data_ptr() on meta tensors is always 0 so they are equal regardless of the reshape
        if device != "meta":
            self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        s = torch.randn((), device=device)
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        self.assertEqual(s.reshape(-1).shape, (1,))
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        empty = torch.tensor([], device=device)
        self.assertEqual(empty, empty.reshape(-1))
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: fix these once we have multi-dimensional empty tensors
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        self.assertRaises(RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device)))

    def test_narrow(self, device="mps"):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))

    def test_narrow_tensor(self, device="mps"):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    def test_t(self, device="mps"):
        # Test 0D tensors
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 1D tensors
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 2D tensors
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # Test 3D tensor
        x = torch.rand((2, 2, 2))
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 dimensions, but self is 3D'):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 sparse and 0 dense dimensions'):
            x.t()

    def test_split(self, device="mps"):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        # Variable sections split
        tensor = torch.randn(20, 10)
        dim = 0
        split_sizes = [5, 5, 10]
        target_sizes = ([[5, 10], [5, 10], [10, 10]])
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        split_sizes = [2, 2, 6]
        target_sizes = ([20, 2], [20, 2], [20, 6])
        dim = 1
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

    def test_chunk(self, device="mps"):
        tensor = torch.rand(4, 7)
        num_chunks = 3
        dim = 1
        target_sizes = ([4, 3], [4, 3], [4, 1])
        splits = tensor.chunk(num_chunks, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split,
                             atol=0, rtol=0)
            start = start + target_size[dim]

        # Invalid chunk sizes
        error_regex = 'chunk expects.*greater than 0'
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(0)
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(-2)

    def test_unsqueeze(self, device="mps") -> None:
        x = torch.randn(2, 3, 4)
        y = x.unsqueeze(1)
        self.assertEqual(y, x.view(2, 1, 3, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.view(2, 3, 1, 4))

        x = x[:, 1]
        self.assertFalse(x.is_contiguous())
        y = x.unsqueeze(1)
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

    # unit test for special case transposed copy (see ATen/native/Copy.cpp for details)
    def test_big_transpose(self, device="mps"):
        t = torch.rand(456, 789, device=device)
        t1 = t.t().contiguous()
        t2 = torch.from_numpy(t.cpu().numpy().transpose())
        self.assertEqual(t1, t2)

    def test_T(self, device="mps"):
        a = torch.randn(2, 3, 4, device=device)
        t1 = a.T
        t2 = a.permute(2, 1, 0)
        self.assertEqual(t2, t1)
        b = torch.randn(10, device=device)
        self.assertEqual(b, b.T)

    def test_transposes(self, device="mps", dtype=torch.float32):
        for op in ("T", "H", "mT", "mH", "adjoint"):
            shapes = ((2, 3), (2, 3, 4)) if op[0] == "m" or op == "adjoint" else ((2, 3),)
            for shape in shapes:
                a = make_tensor(shape, device=device, dtype=dtype)
                t1 = getattr(a, op)
                if op == "adjoint":
                    t1 = t1()
                t2 = a
                if a.ndim != 0:
                    t2 = t2.transpose(-2, -1)
                if op[-1] == "H" or op == "adjoint":
                    t2 = t2.conj()
                self.assertEqual(t2, t1)

    def test_transposes_errors(self, device="mps", dtype=torch.float32):
        for op in ("H", "mT", "mH", "adjoint"):
            shapes = ((2,), (2, 3, 4)) if op == "H" else ((2,),)
            for shape in shapes:
                a = make_tensor(shape, device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "only supported on matrices"):
                    t1 = getattr(a, op)
                    if op == "adjoint":
                        t1 = t1()

    def test_python_types(self, device="mps"):
        a1 = torch.randn((1, 2), device=device, dtype=torch.float32)
        a2 = torch.randn((1, 2), device=device, dtype=torch.float32)
        self.assertEqual(a1.dtype, a2.dtype)

        b1 = torch.arange(10, 20, dtype=torch.int64, device=device)
        b2 = torch.arange(10, 20, dtype=int, device=device)
        self.assertEqual(b1.dtype, b2.dtype)

        c1 = torch.tensor([True, False], dtype=torch.bool, device=device)
        c2 = torch.tensor([True, False], dtype=bool, device=device)
        self.assertEqual(c1.dtype, c2.dtype)

    # TODO: is resize best put in test_view_ops?
    def test_resize_as_preserves_strides(self, device="mps"):
        x = torch.empty(2, 3).t()
        old_strides = x.stride()
        x.resize_as_(x)
        self.assertEqual(x.stride(), old_strides)

    def test_memory_format_resize_as(self, device="mps"):
        def test_helper(shape, memory_format, device="mps"):
            xc = torch.randn(shape, device=device).contiguous(memory_format=memory_format)
            flat = torch.randn(xc.numel(), device=device)
            flat.resize_as_(xc, memory_format=torch.preserve_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), torch.channels_last_3d, device="mps")

    def test_memory_format_resize_(self, device="mps"):
        def test_helper(shape, numel, memory_format, device="mps"):
            flat = torch.randn(numel, device=device)
            flat.resize_(shape, memory_format=memory_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), 10 * 3 * 32 * 32, torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), 3 * 10 * 3 * 32 * 32, torch.channels_last_3d, device="mps")

    # TODO: OpInfo this
    def _test_atleast(self, device, torch_fn):
        # 0-dim
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), s)
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1-dim
        a = torch.rand(4, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), a)
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2,3,4-dim
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    def test_atleast_gradient(self, device="mps"):
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)

    def test_view(self, device="mps"):
        tensor = torch.rand(15, device=device)
        template = torch.rand(3, 5, device=device)
        empty = torch.empty(0, device=device)
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # test size inference with empty tensors
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

    def test_contiguous(self, device="mps"):
        x = torch.randn(1, 16, 5, 5, device=device)
        self.assertTrue(x.is_contiguous())
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())

    def test_resize_mps_dtypes(self, device="mps"):
        shape = (2, 2)
        for dt in MPS_DTYPES:
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            x.resize_(shape)
            self.assertEqual(shape, x.shape)

    def test_resize_as_mps_dtypes(self, device="mps"):
        for dt in MPS_DTYPES:
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dt, device=device)
            x.resize_as_(y)
            self.assertEqual(y.shape, x.shape)

    def test_resize_overflow(self, device="mps"):
        x = torch.empty((), dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            x.resize_([2, 4, 2**29, 2**29])
        with self.assertRaisesRegex(RuntimeError, 'overflow'):
            x.resize_([8, 8, 2**29, 2**29])

    def test_view_all_dtypes_and_devices(self, device="mps"):
        for dt in (torch.float, torch.bool):
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            self.assertEqual(x.view(6).shape, [6])

class TestConvolutionMPS(TestCaseMPS):
    def test_conv1d_all_strides_paddings(self):
        # https://github.com/pytorch/pytorch/issues/82921
        def helper(stride, padding):
            y_cpu = torch.randn(1, 57, 40)
            conv_cpu = nn.Conv1d(57, 20, stride=stride, padding=padding, kernel_size=3, bias=False)
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            x_cpu = conv_cpu(y_cpu)

            y_gpu = y_cpu.to(device='mps')
            x_gpu = conv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu())
        for stride in range(1, 4):
            for padding in range(1, 4):
                helper(stride, padding)


    def test_conv1d_channels_last(self):
        # https://github.com/pytorch/pytorch/issues/81557
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        a_cpu = torch.arange((128 * 176), dtype=torch.float32)
        a_cpu = a_cpu.view(128, 176, 1).permute(0, 2, 1)
        out_cpu = model_cpu(a_cpu)

        a_mps = a_cpu.detach().clone().to("mps")
        model_mps = model_cpu.to("mps")
        out_mps = model_mps(a_mps)

        self.assertEqual(out_cpu, out_mps.cpu(), rtol=2.6e-05, atol=2e-04)

    def test_conv_transpose_1d_all_strides(self):
        # https://github.com/pytorch/pytorch/issues/82711
        def helper(stride):
            y_cpu = torch.ones(1, 1, 2)
            deconv_cpu = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1, stride=stride, bias=False, padding=1)
            deconv_cpu.weight.data = torch.ones(1, 1, 2)
            deconv_gpu = copy.deepcopy(deconv_cpu).to(device='mps')
            x_cpu = deconv_cpu(y_cpu)

            y_gpu = y_cpu.to(device='mps')
            x_gpu = deconv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu())
        [helper(stride) for stride in [1, 2, 3]]

    def test_conv_transpose_1d_nn_functional(self):
        # https://github.com/pytorch/pytorch/issues/82563
        tin = torch.rand((1, 512, 1245), dtype=torch.float32)
        tparams = torch.rand((512, 256, 16), dtype=torch.float32)
        tbias = torch.rand((256), dtype=torch.float32)

        device = 'cpu'
        tcpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        device = 'mps'
        tgpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        self.assertEqual(tcpu, tgpu.cpu(), rtol=2.6e-05, atol=2e-04)

    def test_conv_backward_1d_channels_last(self):
        def helper(shape, in_channels=1, out_channels=1, kernel_size=3, groups=1):
            # https://github.com/pytorch/pytorch/issues/84511
            conv_cpu = torch.nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups).requires_grad_()
            conv_mps = torch.nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups).to("mps")
            conv_mps.weight.data = conv_cpu.weight.data.detach().clone().to("mps").requires_grad_(True)
            conv_mps.bias.data = conv_cpu.bias.data.detach().clone().to("mps").requires_grad_(True)


            data = torch.rand(shape, dtype=torch.float32)
            x_cpu = data.permute(0, 2, 1).contiguous().requires_grad_(True)
            x_mps = data.permute(0, 2, 1).detach().clone().to("mps").contiguous().requires_grad_(True)
            res_cpu = conv_cpu(x_cpu)
            res_mps = conv_mps(x_mps)
            self.assertEqual(res_cpu, res_mps)
            res_cpu = res_cpu.sum().backward()
            res_mps = res_mps.sum().backward()

            self.assertEqual(conv_cpu.weight.grad, conv_mps.weight.grad, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x_cpu.grad, x_mps.grad)

        helper(shape=(1, 176, 1))
        helper(shape=(2, 12, 1))
        helper(shape=(3, 176, 1))
        helper(shape=(4, 376, 1))
        helper(shape=(1024, 376, 9), in_channels=9, out_channels=1, groups=1)
        helper(shape=(1024, 376, 9), in_channels=9, out_channels=9, groups=3)

    def test_conv1d_contiguous(self):
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        a_cpu = torch.ones(128, 1, 176)
        out_cpu = model_cpu(a_cpu)

        a_mps = a_cpu.detach().clone().to("mps")
        model_mps = model_cpu.to("mps")
        out_mps = model_mps(a_mps)

        self.assertEqual(out_cpu.shape, out_mps.shape)
        self.assertEqual(out_cpu, out_mps.cpu())

    def test_conv2d_all_strides_paddings(self):
        # https://github.com/pytorch/pytorch/issues/83180
        def helper(N, C, H, W, groups, input_mem_format, weight_mem_format, permute_data):
            x_cpu = torch.randn(N, C, H, W).to(memory_format=input_mem_format).requires_grad_()
            x_mps = x_cpu.detach().clone().to(device='mps').requires_grad_()

            if permute_data:
                x_cpu.permute(0, 2, 3, 1)
                x_mps.permute(0, 2, 3, 1)

            for strideX in range(1, 4):
                for strideY in range(1, 4):
                    conv_cpu = torch.nn.Conv2d(
                        in_channels=N, out_channels=C, kernel_size=H, groups=groups, stride=(strideX, strideY)).requires_grad_()
                    conv_cpu.weight.data = conv_cpu.weight.to(memory_format=weight_mem_format).requires_grad_()

                    conv_mps = torch.nn.Conv2d(
                        in_channels=N, out_channels=C, kernel_size=H, groups=groups, stride=(strideX, strideY), device="mps")
                    conv_mps.weight.data = conv_cpu.weight.data.detach().clone().to("mps").requires_grad_()
                    conv_mps.bias.data = conv_cpu.bias.data.detach().clone().to("mps").requires_grad_()

                    res_cpu = conv_cpu(x_cpu)
                    res_mps = conv_mps(x_mps)
                    self.assertEqual(res_cpu, res_mps.cpu(), rtol=1e-03, atol=1e-05)

                    res_cpu = res_cpu.sum().backward()
                    res_mps = res_mps.sum().backward()
                    self.assertEqual(res_cpu, res_mps, rtol=2.6e-05, atol=2e-04)
                    self.assertEqual(conv_cpu.weight.grad, conv_mps.weight.grad, rtol=2.6e-05, atol=2e-04)
                    self.assertEqual(conv_cpu.bias.grad, conv_mps.bias.grad)
                    self.assertEqual(x_cpu.grad, x_mps.grad)

        for mem_format_input in [torch.contiguous_format, torch.channels_last]:
            for mem_format_weight in [torch.contiguous_format, torch.channels_last]:
                for permute_data in [True, False]:
                    helper(2, 2, 3, 6, 1, mem_format_input, mem_format_weight, permute_data)
                    helper(10, 10, 4, 6, 2, mem_format_input, mem_format_weight, permute_data)
                    helper(32, 32, 4, 6, 2, mem_format_input, mem_format_weight, permute_data)

    def test_conv_transpose_2d_strided(self):
        def helper(m_cpu, memory_format):
            m_mps = copy.deepcopy(m_cpu).requires_grad_()
            m_mps.weight.data = m_cpu.weight.data.detach().clone().to("mps").requires_grad_()
            m_mps.bias.data = m_cpu.bias.data.detach().clone().to("mps").requires_grad_()

            input_cpu = torch.randn(20, 16, 50, 100).to(memory_format=memory_format).requires_grad_()
            input_mps = input_cpu.detach().clone().to("mps")

            output_cpu = m_cpu(input_cpu)
            output_mps = m_mps(input_mps)
            self.assertEqual(output_cpu, output_mps)

        for mem_format_input in [torch.contiguous_format, torch.channels_last]:
            # With square kernels and equal stride
            helper(nn.ConvTranspose2d(16, 33, 3, stride=2).requires_grad_(), mem_format_input)

            # non-square kernels and unequal stride and with padding
            helper(nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)).requires_grad_(), mem_format_input)

    def test_conv_transpose_2d_specified_output(self):
        input_cpu = torch.randn(1, 16, 12, 12)
        input_mps = input_cpu.detach().clone().to("mps")

        downsample_cpu = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        downsample_mps = nn.Conv2d(16, 16, 3, stride=2, padding=1, device="mps")
        downsample_mps.weight.data = downsample_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        downsample_mps.bias.data = downsample_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        upsample_cpu = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        upsample_mps = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, device="mps")
        upsample_mps.weight.data = upsample_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        upsample_mps.bias.data = upsample_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        h_cpu = downsample_cpu(input_cpu)
        h_mps = downsample_mps(input_mps)
        self.assertEqual(h_cpu, h_mps)

        size_cpu = h_cpu.size()
        size_mps = h_mps.size()
        self.assertEqual(size_cpu, size_mps)

        output_cpu = upsample_cpu(h_cpu, output_size=input_cpu.size())
        output_mps = upsample_mps(h_mps, output_size=input_mps.size())
        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_conv2d_single_stride(self):
        y_cpu = torch.randn(2, 2, 3, 6)
        y_gpu = y_cpu.to(device='mps')
        for stride in range(1, 4):
            conv_cpu = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=stride)
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            x_cpu = conv_cpu(y_cpu)
            x_gpu = conv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu(), rtol=1e-03, atol=1e-05)

    def test_grid_sample(self):
        def test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad):
            def test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners):
                for grid_dim_contig_order in [(0, 1, 2, 3), (0, 3, 1, 2), (3, 0, 1, 2), (0, 2, 1, 3)]:
                    # grid_dim_contig_order specifies the dimension order that can
                    # make grid to be contiguous.
                    # i.e., grid.permute(grid_dim_contig_order) is contiguous.
                    # e.g., with grid_dim_contig_order=[0, 3, 1, 2], grid should be
                    #       initialized with contiguous tensor of shape [N, 2, H, W]
                    #       and permuted to [N, H, W, 2] afterwards.
                    grid_shape = [N, H, W, 2]
                    grid_init_shape = [grid_shape[d] for d in grid_dim_contig_order]
                    grid_fwd_permute = [None, None, None, None]
                    for i, d in enumerate(grid_dim_contig_order):
                        grid_fwd_permute[d] = i

                    def get_grid(device='cpu', data=None):
                        if data is not None:
                            assert list(data.shape) == grid_shape
                            data = data.permute(grid_dim_contig_order).to(device)
                        else:
                            data = torch.randn(grid_init_shape, device=device)
                        grid = data.permute(grid_fwd_permute)
                        assert grid.permute(grid_dim_contig_order).is_contiguous()
                        return grid

                    input_cpu = torch.randn(C, N, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_cpu = get_grid().requires_grad_()
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                    self.assertTrue(out_cpu.size() == torch.Size([N, C, H, W]))

                    gradients = torch.randn_like(out_cpu)
                    out_cpu.backward(gradients)


                    # Compare against unvectorized CPU fallback

                    # NOTE [ grid_sample CPU fallback ]
                    # grid_sample uses AVX for 2d images, but that requires 32-bit indexing for
                    # 32-bit floats. So we also have a fallback that is used only for float tensors
                    # requiring 64-bit indexing. That requires too much memory to run on CI, so we
                    # also export the fallback and test it here to ensure feature parity with
                    # the vectorized version.
                    input_fallback = input_cpu.float().detach_().requires_grad_()
                    grid_fallback = grid_cpu.float().detach_().requires_grad_()
                    out_fallback = torch._grid_sampler_2d_cpu_fallback(
                        input_fallback, grid_fallback,
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners)
                    self.assertEqual(out_fallback, out_cpu.float(), atol=1e-5, rtol=5e-5)

                    out_fallback.backward(gradients.float())
                    if input_requires_grad:
                        self.assertEqual(input_fallback.grad, input_cpu.grad.float(), atol=1e-4, rtol=5e-5)
                    self.assertEqual(grid_fallback.grad, grid_cpu.grad.float(), atol=1e-4, rtol=5e-5)

                    input_mps = input_cpu.detach().transpose(0, 1).to("mps").transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_mps = get_grid('mps', grid_cpu.detach()).requires_grad_()
                    out_mps = F.grid_sample(input_mps, grid_mps, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    self.assertEqual(out_cpu, out_mps)
                    out_mps.backward(gradients.to("mps"))
                    if input_requires_grad:
                        self.assertEqual(input_cpu.grad, input_mps.grad)
                    self.assertEqual(grid_cpu.grad, grid_mps.grad, atol=5e-5, rtol=0)

                    # check that zero-dimensional input strides don't error out
                    base_input = torch.randn(N, C, 1, IW)
                    input_cpu = base_input.expand_as(input_mps).requires_grad_(input_requires_grad)
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)

                    input_mps = base_input.to("mps").expand_as(input_mps).requires_grad_(input_requires_grad)
                    out_mps = F.grid_sample(input_mps, grid_mps, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    self.assertEqual(out_cpu, out_mps)

            # test same size output
            test_shape(N, C, H, W, H, W, mode, padding_mode, align_corners)

            # test larger output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(IH + 1, 12)
            W = random.randint(IW + 1, 12)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test smaller output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(2, IH)
            W = random.randint(2, IW)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test 1x1 inpput
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = 1
            IW = 1
            H = random.randint(2, 5)
            W = random.randint(2, 5)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty grid
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            W = random.randint(3, IW + 2)
            test_shape(N, C, IH, IW, 0, W, mode, padding_mode, align_corners)

            # testing empty channel
            N = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, 0, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty batch
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(0, C, IH, IW, H, W, mode, padding_mode, align_corners)

        for mode in ('bilinear', 'nearest'):
            for padding_mode in ('zeros', 'reflection'):
                for align_corners in (True, False):
                    # test known input
                    input = torch.arange(1., 11, device="mps").view(1, 1, 2, 5)
                    grid = torch.tensor(
                        [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
                         [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]], device="mps").view(1, 2, 5, 2)
                    if mode == 'bilinear':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[0.0000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 0.0000]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.0000, 6.5000000000, 1.2500, 4.6675000191, 4.6250],
                                     [0.5000, 7.1665000916, 1.2500, 5.0000000000, 0.0000]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1.2000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 8.7500]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1.0000, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000, 7.1665000916, 5.0000, 5.0000000000, 10.0000]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[3.4500, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 7.7500]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[3.0000004768, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000000000, 7.1665000916, 5.0000, 5.0000000000, 9.2500]], device="mps").view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                    elif mode == 'nearest':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[0., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 0.]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0., 8., 5., 7., 0.],
                                     [1., 8., 5., 8., 0.]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]], device="mps").view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                    elif mode == 'bicubic':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[-0.10424726, 7.1400003, 5.0000, 5.7842274, 9.0000],
                                     [2.4492188, 7.4814040, 5.0000, 6.0277520, 0.0000]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.00000, 7.6287503, 1.0625, 5.5977230, 5.3270264],
                                     [0.40625, 8.0288770, 1.0625, 5.9375067, -0.3515625]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1.1520010, 6.0599990, 5.0000, 4.870930, 9.0000000],
                                     [2.1328125, 6.4258375, 5.0000, 5.076003, 8.8671875]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.894531, 6.6050020, 4.625, 4.7138715, 9.800781],
                                     [0.906250, 7.2822485, 4.625, 5.0000052, 10.00000]], device="mps").view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[3.1822524, 6.239998, 5.0000, 4.8709273, 9.00000],
                                     [1.7812500, 6.703594, 5.0000, 5.0760007, 8.21875]], device="mps").view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[2.7993753, 6.6050020, 4.25, 4.7138715, 10.269531],
                                     [0.8125000, 7.2822485, 4.25, 5.0000052, 9.332031]], device="mps").view(1, 1, 2, 5)
                        else:
                            raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")

                    else:
                        raise AssertionError(f"missing groundtruth test for interpolation mode '{mode}'")
                    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                           align_corners=align_corners)
                    self.assertEqual(output, groundtruth, atol=1e-5, rtol=0,
                                     msg=f"groundtruth comparison failed for mode={mode}, "
                                     f"padding_mode={padding_mode}")

class TestAdvancedIndexing(TestCaseMPS):
    supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
    supported_np_dtypes = [np.float32, np.float16, np.int64, np.int32, np.int16, np.uint8]

    def test_nonzero_no_warning(self):
        device = "mps"
        t = torch.randn((2, 2), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.nonzero(t)
            t.nonzero()
            self.assertEqual(len(w), 0)

    def test_nonzero(self):
        def helper(dtype):
            device = "mps"
            shapes = [
                torch.Size((12,)),
                torch.Size((12, 1)),
                torch.Size((1, 12)),
                torch.Size((6, 2)),
                torch.Size((3, 2, 2)),
                torch.Size((5, 5, 5)),
            ]

            def gen_nontrivial_input(shape, dtype, device):
                if dtype != torch.bfloat16:
                    return torch.randint(2, shape, device=device, dtype=dtype)
                else:
                    # windows does not work for bfloat16 randing
                    return torch.randint(2, shape, device=device, dtype=torch.float).to(dtype)

            for shape in shapes:
                tensor = gen_nontrivial_input(shape, dtype, device)
                dst1 = torch.nonzero(tensor, as_tuple=False)
                dst2 = tensor.nonzero(as_tuple=False)
                dst3 = torch.empty([], dtype=torch.long, device=device)
                dst3 = dst3.resize_(0)
                torch.nonzero(tensor, out=dst3)
                np_array = tensor.cpu().numpy() if dtype != torch.bfloat16 else tensor.float().cpu().numpy()
                np_result = torch.from_numpy(np.stack(np_array.nonzero())).t()
                self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
                self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
                self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
                tup1 = torch.nonzero(tensor, as_tuple=True)
                tup2 = tensor.nonzero(as_tuple=True)
                tup1 = torch.stack(tup1).t().cpu()
                tup2 = torch.stack(tup2).t().cpu()
                self.assertEqual(tup1, np_result, atol=0, rtol=0)
                self.assertEqual(tup2, np_result, atol=0, rtol=0)
        [helper(dtype) for dtype in self.supported_dtypes]

    def test_nonzero_astuple_out(self):
        device = "mps"
        t = torch.randn((3, 3, 3), device=device)
        out = torch.empty([], dtype=torch.long, device=device)
        out = out.resize_(0)

        with self.assertRaises(RuntimeError):
            torch.nonzero(t, as_tuple=True, out=out)

        self.assertEqual(torch.nonzero(t, as_tuple=False, out=out), torch.nonzero(t, out=out))

        # Verifies that JIT script cannot handle the as_tuple kwarg
        # See Issue https://github.com/pytorch/pytorch/issues/45499.
        def _foo(t):
            tuple_result = torch.nonzero(t, as_tuple=True)
            nontuple_result = torch.nonzero(t, as_tuple=False)
            out = torch.empty_like(nontuple_result)
            torch.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        with self.assertRaises(RuntimeError):
            scripted_foo = torch.jit.script(_foo)

        # Verifies that JIT tracing works fine
        traced_foo = torch.jit.trace(_foo, t)
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        expected_tuple = torch.nonzero(t, as_tuple=True)
        expected_nontuple = torch.nonzero(t)

        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    def test_nonzero_discontiguous(self):
        device = "mps"
        shape = (4, 4)
        tensor = torch.randint(2, shape, device=device)
        tensor_nc = torch.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(tensor)
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        dst3 = torch.empty_like(dst1)
        data_ptr = dst3.data_ptr()
        # expect dst3 storage to be reused
        torch.nonzero(tensor, out=dst3)
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        # discontiguous out
        dst4 = torch.empty(dst1.size(0), dst1.size(1) * 2, dtype=torch.long, device=device)[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        torch.nonzero(tensor, out=dst4)
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    def test_nonzero_non_diff(self):
        device = "mps"
        x = torch.randn(10, requires_grad=True)
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

    def test_nonzero_multi_threading(self):
        # Test that MPS does not crash if nonzero called concurrently
        # See https://github.com/pytorch/pytorch/issues/100285
        x = torch.rand(3, 3, device="mps")
        t1 = threading.Thread(target=torch.nonzero, args=(x,))
        t2 = threading.Thread(target=torch.nonzero, args=(x,))
        t1.start()
        t2.start()

    def test_masked_select(self):
        x = torch.randn(3, 4)
        x_mps = x.to("mps")
        mask = x.ge(0.5)
        mask_mps = x_mps.ge(0.5)

        res = torch.masked_select(x, mask)
        res_mps = torch.masked_select(x_mps, mask_mps)

        self.assertEqual(res, res_mps)

    # examples from https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
    def test_indexing_get(self):
        def helper(dtype):
            x_cpu = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            y_cpu = x_cpu[[0, 1, 2], [0, 1, 0]]
            y_mps = x_mps[[0, 1, 2], [0, 1, 0]]
            self.assertEqual(y_cpu, y_mps, str(dtype))
        [helper(dtype) for dtype in self.supported_dtypes]

    def test_indexing_select_corners(self):
        def helper(dtype):
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            rows_cpu = torch.tensor([[0, 0], [3, 3]])
            rows_mps = rows_cpu.detach().clone().to("mps")

            cols_cpu = torch.tensor([[0, 2], [0, 2]])
            cols_mps = cols_cpu.detach().clone().to("mps")

            res_cpu = x_cpu[rows_cpu, cols_cpu]
            res_mps = x_mps[rows_mps, cols_mps]

            self.assertEqual(res_cpu, res_mps, str(dtype))
        [helper(dtype) for dtype in self.supported_dtypes]

    # FIXME: uint8 fails for this testcase, needs further debugging
    def test_slicing_using_advanced_index_for_column(self):
        def helper(dtype):
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            z_cpu = x_cpu[1:4, 1:3]
            z_mps = x_mps[1:4, 1:3]
            self.assertEqual(z_cpu, z_mps, str(dtype))

            # using advanced index for column
            y_cpu = x_cpu[1:4, [1, 2]]
            y_mps = x_mps[1:4, [1, 2]]
            self.assertEqual(y_cpu, y_mps, str(dtype))
        # FIXME: use supported_dtypes once uint8 is fixed
        [helper(dtype) for dtype in [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16]]

    def test_boolean_array_indexing(self):
        def helper(dtype):
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            res_cpu = x_cpu[x_cpu > 5]
            res_mps = x_mps[x_mps > 5]

            self.assertEqual(res_cpu, res_mps, str(dtype))
        for dtype in self.supported_dtypes:
            # MPS support binary op with uint8 natively starting from macOS 13.0
            if product_version < 13.0 and dtype == torch.uint8:
                continue
            helper(dtype)

    def test_advanced_indexing_3D_get(self):
        def helper(x_cpu):
            x_mps = x_cpu.detach().clone().to("mps")
            self.assertEqual(x_cpu[[1, 2], 3, :], x_mps[[1, 2], 3, :])
            self.assertEqual(x_cpu[[0, 2], :, :], x_mps[[0, 2], :, :])
            self.assertEqual(x_cpu[:, [1, 0], [1]], x_mps[:, [1, 0], [1]])

        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        helper(x_cpu)
        for idx in range(len(self.supported_np_dtypes)):
            # torch.randn / torch.rand don't work with all dtypes
            # Generate input data for all dtypes on Numpy them move to torch
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            helper(inputCPU)

    def test_advanced_indexing_3D_put(self):
        def helper(x_cpu):
            dtype = x_cpu.dtype
            x_mps = x_cpu.detach().clone().to("mps")

            out_tensor_cpu = torch.tensor([88, 99], dtype=dtype, device="cpu")
            out_tensor_cpu_view = out_tensor_cpu[1:]

            out_tensor_mps = torch.tensor([88, 99], dtype=dtype, device="mps")
            out_tensor_mps_view = out_tensor_mps[1:]

            x_cpu[[1, 2], 3, :] = out_tensor_cpu_view
            x_mps[[1, 2], 3, :] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

            x_cpu[[0, 2], :, :] = out_tensor_cpu_view
            x_mps[[0, 2], :, :] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

            x_cpu[:, [1, 0], [1]] = out_tensor_cpu_view
            x_mps[:, [1, 0], [1]] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        helper(x_cpu)
        for idx in range(len(self.supported_np_dtypes)):
            # torch.randn / torch.rand don't work with all dtypes
            # Generate input data for all dtypes on Numpy them move to torch
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            helper(inputCPU)

    def test_index_put_with_view_indices(self):
        def helper(dtype):
            target_cpu = torch.zeros([5, 3], device="cpu", dtype=dtype)
            target_mps = torch.zeros([5, 3], device="mps", dtype=dtype)

            indices_cpu = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="cpu")
            indices_mps = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="mps")

            value_cpu = torch.ones(indices_cpu.shape[0], device="cpu", dtype=dtype)
            value_mps = torch.ones(indices_mps.shape[0], device="mps", dtype=dtype)

            target_cpu.index_put_(tuple(indices_cpu.t()), value_cpu, accumulate=True)
            target_mps.index_put_(tuple(indices_mps.t()), value_mps, accumulate=True)

            self.assertEqual(target_cpu, target_mps)

        [helper(dtype) for dtype in [torch.int32, torch.float]]

    # tests from 'test_indexing.py'
    def test_advancedindex_big(self, device="mps"):
        reference = torch.arange(0, 123344, dtype=torch.int, device=device)

        self.assertEqual(reference[[0, 123, 44488, 68807, 123343], ],
                         torch.tensor([0, 123, 44488, 68807, 123343], dtype=torch.int))

    def test_set_item_to_scalar_tensor(self, device="mps"):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        z = torch.randn([m, n], device=device)
        a = 1.0
        w = torch.tensor(a, requires_grad=True, device=device)
        z[:, 0] = w
        z.sum().backward()
        self.assertEqual(w.grad, m * a)

    def test_single_int(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))

    def test_multiple_int(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))

    def test_none(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

    def test_step(self, device="mps"):
        v = torch.arange(10, device=device)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])

    def test_step_assignment(self, device="mps"):
        v = torch.zeros(4, 4, device=device)
        v[0, 1::2] = torch.tensor([3., 4.], device=device)
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)

    def test_bool_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=device)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool, device=device)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], torch.tensor([True], dtype=torch.bool, device=device))
            self.assertEqual(len(w), 2)

    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    def test_bool_indices_accumulate(self, device="mps"):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        mask = mask > 0
        y = torch.ones(size=(10, 10), device=device)
        y.index_put_((mask, ), y[mask], accumulate=True)
        self.assertEqual(y, torch.ones(size=(10, 10), device=device))

    def test_multiple_bool_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool, device=device)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    def test_byte_mask(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        mask = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[mask].shape, (3, 7, 3))
            self.assertEqual(v[mask], torch.stack([v[0], v[2], v[3]]))
            self.assertEqual(len(w), 2)

        v = torch.tensor([1.], device=device)
        self.assertEqual(v[v == 0], torch.tensor([], device=device))

    def test_byte_mask_accumulate(self, device="mps"):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        y = torch.ones(size=(10, 10), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y.index_put_((mask, ), y[mask], accumulate=True)
            self.assertEqual(y, torch.ones(size=(10, 10), device=device))
            self.assertEqual(len(w), 2)

    def test_index_put_accumulate_expanded_values(self, device="mps"):
        t = torch.zeros((5, 2))
        t_dev = t.to(device)
        indices = [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([1, ]),
        ]
        indices_dev = [i.to(device) for i in indices]
        values0d = torch.tensor(1.0)
        values1d = torch.tensor([1.0, ])

        out_mps = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values0d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        t = torch.zeros(4, 3, 2)
        t_dev = t.to(device)

        indices = [
            torch.tensor([0, ]),
            torch.arange(3)[:, None],
            torch.arange(2)[None, :],
        ]
        indices_dev = [i.to(device) for i in indices]
        values1d = torch.tensor([-1.0, -2.0])
        values2d = torch.tensor([[-1.0, -2.0], ])

        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values2d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_non_contiguous(self, device="mps"):
        t = torch.zeros((5, 2, 2))
        t_dev = t.to(device)
        t1 = t_dev[:, 0, :]
        t2 = t[:, 0, :]
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        indices = [torch.tensor([0, 1]), ]
        indices_dev = [i.to(device) for i in indices]
        value = torch.randn(2, 2)
        out_mps = t1.index_put_(indices_dev, value.to(device), accumulate=True)
        out_cpu = t2.index_put_(indices, value, accumulate=True)
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_with_optional_tensors(self, device="mps"):
        # TODO: replace with a better solution.
        # Currently, here using torchscript to put None into indices.
        # on C++ it gives indices as a list of 2 optional tensors: first is null and
        # the second is a valid tensor.
        @torch.jit.script
        def func(x, i, v):
            idx = [None, i]
            x.index_put_(idx, v, accumulate=True)
            return x

        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])

        out_mps = func(t_dev, indices_dev, value0d.to("mps"))
        out_cpu = func(t, indices, value0d)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = func(t_dev, indices_dev, value1d.to("mps"))
        out_cpu = func(t, indices, value1d)
        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_duplicate_indices(self, device="mps"):
        for i in range(1, 128):
            # generate indices by random walk, this will create indices with
            # lots of duplicates interleaved with each other
            delta = torch.empty(i, dtype=torch.float32, device=device).uniform_(-1, 1)

            indices = delta.cumsum(0).long().to("mps")

            # abs for int64 is not supported on mps, fallback on 'cpu' to calculate it
            input = torch.randn(indices.cpu().abs().max().to("mps") + 1, device=device)
            values = torch.randn(indices.size(0), device=device)
            output = input.index_put((indices,), values, accumulate=True)

            input_list = input.tolist()
            indices_list = indices.tolist()
            values_list = values.tolist()
            for i, v in zip(indices_list, values_list):
                input_list[i] += v

            self.assertEqual(output, input_list)

    def test_index_put_deterministic(self, device="mps"):
        def helper(dtype, accumulate, deterministic, num_tests=128):
            acc_expected = torch.tensor([233, 187, 360], device=device, dtype=dtype)
            non_acc_expected = torch.tensor([38, 37, 39], device=device, dtype=dtype)
            t_idx = torch.tensor(
                [0, 0, 0, 0, 2, 2, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2,
                 0, 0, 2, 1, 2, 1, 0, 0, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 0, 2]
            )
            for _ in range(num_tests):
                try:
                    torch.use_deterministic_algorithms(deterministic)
                    t = torch.zeros(3, dtype=dtype, device=device)
                    t.index_put_((t_idx,), torch.arange(len(t_idx), device=device, dtype=dtype), accumulate=accumulate)
                    if accumulate:
                        self.assertEqual(t, acc_expected)
                    else:
                        self.assertEqual(t, non_acc_expected)
                finally:
                    torch.use_deterministic_algorithms(False)

        for accumulate, deterministic in product((False, True), (False, True)):
            dtype = torch.float if accumulate else torch.long
            if not accumulate and not deterministic:
                with self.assertRaisesRegex(AssertionError, "Tensor-likes are not equal!"):
                    helper(dtype, accumulate, deterministic)
            else:
                helper(dtype, accumulate, deterministic)

    def test_multiple_byte_mask(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        mask2 = torch.ByteTensor([1, 1, 1]).to(device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
            self.assertEqual(len(w), 2)

    def test_byte_mask2d(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        c = torch.randn(5, 7, device=device)
        num_ones = (c > 0).sum()
        r = v[c > 0]
        self.assertEqual(r.shape, (num_ones, 3))

    def test_jit_indexing(self, device="mps"):
        def fn1(x):
            x[x < 50] = 1.0
            return x

        def fn2(x):
            x[0:50] = 1.0
            return x

        scripted_fn1 = torch.jit.script(fn1)
        scripted_fn2 = torch.jit.script(fn2)
        data = torch.arange(100, device=device, dtype=torch.float)
        out = scripted_fn1(data.detach().clone())
        ref = torch.tensor(np.concatenate((np.ones(50), np.arange(50, 100))), device=device, dtype=torch.float)
        self.assertEqual(out, ref)
        out = scripted_fn2(data.detach().clone())
        self.assertEqual(out, ref)

    def test_int_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

    def test_index_put_src_datatype(self):
        def helper(device, dtype):
            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
            indices = (torch.tensor([0, 2, 1]),)
            res = src.index_put_(indices, vals, accumulate=True)
            self.assertEqual(res.shape, src.shape)
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.int32]]

    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    def test_index_src_datatype(self):
        def helper(device, dtype):
            orig_dtype = dtype
            if dtype is torch.bool:
                dtype = torch.uint8

            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            if orig_dtype is torch.bool:
                src = src == 1
            # test index
            res = src[[0, 2, 1], :, :]
            self.assertEqual(res.shape, src.shape)
            # test index_put, no accum
            src[[0, 2, 1], :, :] = res
            self.assertEqual(res.shape, src.shape)
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.float16, torch.long, torch.bool]]

    def test_int_indices2d(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]], device=device)
        columns = torch.tensor([[0, 2], [0, 2]], device=device)
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])

    def test_int_indices_broadcast(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        rows = torch.tensor([0, 3], device=device)
        columns = torch.tensor([0, 2], device=device)
        result = x[rows[:, None], columns]
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])

    def test_empty_index(self, device="mps"):
        x = torch.arange(0, 12, device=device).view(4, 3)
        idx = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(x[idx].numel(), 0)

        # empty assignment should have no effect but not throw an exception
        y = x.clone()
        y[idx] = -1
        self.assertEqual(x, y)

        mask = torch.zeros(4, 3, device=device).bool()
        y[mask] = -1
        self.assertEqual(x, y)

    def test_empty_ndim_index(self, device="mps"):
        x = torch.randn(5, device=device)
        self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

        x = torch.randn(2, 3, 4, 5, device=device)
        self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                         x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        x = torch.empty(10, 0, device=device)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        self.assertEqual(x[[], []].shape, (0,))
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]

    def test_empty_ndim_index_bool(self, device="mps"):
        x = torch.randn(5, device=device)
        self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])

    def test_empty_slice(self, device="mps"):
        x = torch.randn(2, 3, 4, 5, device=device)
        y = x[:, :, :, 1]
        z = y[:, 1:1, :]
        self.assertEqual((2, 0, 4), z.shape)
        # this isn't technically necessary, but matches NumPy stride calculations.
        self.assertEqual((60, 20, 5), z.stride())
        self.assertTrue(z.is_contiguous())

    def test_index_getitem_copy_bools_slices(self, device="mps"):
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        tensors = [torch.randn(2, 3, device=device), torch.tensor(3., device=device)]

        for a in tensors:
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            self.assertEqual(a.data_ptr(), a[...].data_ptr())

    def test_index_setitem_bools_slices(self, device="mps"):
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        tensors = [torch.randn(2, 3, device=device), torch.tensor(3, device=device)]

        for a in tensors:
            # prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
            # (some of these ops already prefix a 1 to the size)
            neg_ones = torch.ones_like(a) * -1
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            a[True] = neg_ones_expanded
            self.assertEqual(a, neg_ones)
            a[False] = 5
            self.assertEqual(a, neg_ones)
            a[true] = neg_ones_expanded * 2
            self.assertEqual(a, neg_ones * 2)
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            a[None] = neg_ones_expanded * 3
            self.assertEqual(a, neg_ones * 3)
            a[...] = neg_ones_expanded * 4
            self.assertEqual(a, neg_ones * 4)
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5

    def test_index_scalar_with_bool_mask(self, device="mps"):
        a = torch.tensor(1, device=device)
        uintMask = torch.tensor(True, dtype=torch.uint8, device=device)
        boolMask = torch.tensor(True, dtype=torch.bool, device=device)
        self.assertEqual(a[uintMask], a[boolMask])
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

        a = torch.tensor(True, dtype=torch.bool, device=device)
        self.assertEqual(a[uintMask], a[boolMask])
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

    def test_setitem_expansion_error(self, device="mps"):
        true = torch.tensor(True, device=device)
        a = torch.randn(2, 3, device=device)
        # check prefix with  non-1s doesn't work
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # NumPy: ValueError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded

    def test_getitem_scalars(self, device="mps"):
        zero = torch.tensor(0, dtype=torch.int64, device=device)
        one = torch.tensor(1, dtype=torch.int64, device=device)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3, device=device)
        self.assertEqual(a[0], a[zero])
        self.assertEqual(a[0][1], a[zero][one])
        self.assertEqual(a[0, 1], a[zero, one])
        self.assertEqual(a[0, one], a[zero, 1])

        # indexing by a scalar should slice (not copy)
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # scalar indexed with scalar
        r = torch.randn((), device=device)
        with self.assertRaises(IndexError):
            r[:]
        with self.assertRaises(IndexError):
            r[zero]
        self.assertEqual(r, r[...])

    def test_setitem_scalars(self, device="mps"):
        zero = torch.tensor(0, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3, device=device)
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        b = torch.randn(3, device=device)

        a_set_with_number[0] = b
        a_set_with_scalar[zero] = b
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        a[1, zero] = 7.7
        self.assertEqual(7.7, a[1, 0])

        # scalar indexed with scalars
        r = torch.randn((), device=device)
        with self.assertRaises(IndexError):
            r[:] = 8.8
        with self.assertRaises(IndexError):
            r[zero] = 8.8
        r[...] = 9.9
        self.assertEqual(9.9, r)

    def test_basic_advanced_combined(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # Check that it is a copy
        unmodified = x.clone()
        x[1:2, [1, 2]].zero_()
        self.assertEqual(x, unmodified)

        # But assignment should modify the original
        unmodified = x.clone()
        x[1:2, [1, 2]] = 0
        self.assertNotEqual(x, unmodified)

    def test_int_assignment(self, device="mps"):
        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = 5
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = torch.arange(5, 7, device=device)
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])

    def test_byte_tensor_assignment(self, device="mps"):
        x = torch.arange(0., 16, device=device).view(4, 4)
        b = torch.ByteTensor([True, False, True, False]).to(device)
        value = torch.tensor([3., 4., 5., 6.], device=device)

        with warnings.catch_warnings(record=True) as w:
            x[b] = value
            self.assertEqual(len(w), 1)

        self.assertEqual(x[0], value)
        self.assertEqual(x[1], torch.arange(4., 8, device=device))
        self.assertEqual(x[2], value)
        self.assertEqual(x[3], torch.arange(12., 16, device=device))

    def test_variable_slicing(self, device="mps"):
        x = torch.arange(0, 16, device=device).view(4, 4)
        indices = torch.IntTensor([0, 1]).to(device)
        i, j = indices
        self.assertEqual(x[i:j], x[0:1])

    def test_ellipsis_tensor(self, device="mps"):
        x = torch.arange(0, 9, device=device).view(3, 3)
        idx = torch.tensor([0, 2], device=device)
        self.assertEqual(x[..., idx].tolist(), [[0, 2],
                                                [3, 5],
                                                [6, 8]])
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2],
                                                [6, 7, 8]])

    def test_invalid_index(self, device="mps"):
        x = torch.arange(0, 16, device=device).view(4, 4)
        self.assertRaisesRegex(TypeError, 'slice indices', lambda: x["0":"1"])

    def test_out_of_bound_index(self, device="mps"):
        x = torch.arange(0, 100, device=device).view(2, 5, 10)
        self.assertRaisesRegex(IndexError, 'index 5 is out of bounds for dimension 1 with size 5', lambda: x[0, 5])
        self.assertRaisesRegex(IndexError, 'index 4 is out of bounds for dimension 0 with size 2', lambda: x[4, 5])
        self.assertRaisesRegex(IndexError, 'index 15 is out of bounds for dimension 2 with size 10',
                               lambda: x[0, 1, 15])
        self.assertRaisesRegex(IndexError, 'index 12 is out of bounds for dimension 2 with size 10',
                               lambda: x[:, :, 12])

    def test_zero_dim_index(self, device="mps"):
        x = torch.tensor(10, device=device)
        self.assertEqual(x, x.item())

        def runner():
            print(x[0])
            return x[0]

        self.assertRaisesRegex(IndexError, 'invalid index', runner)

    def test_cpu_indices(self, device="mps"):
        idx = torch.tensor([0, 1])
        b = torch.zeros(2, device=device)
        x = torch.ones(10, device=device)
        x[idx] = b  # index_put_
        ref = torch.ones(10, device=device)
        ref[:2] = 0
        self.assertEqual(x, ref, atol=0, rtol=0)
        out = x[idx]  # index
        self.assertEqual(out, torch.zeros(2, device=device), atol=0, rtol=0)

class TestRNNMPS(TestCaseMPS):
    def _lstm_helper(self, num_layers, dtype, device, bidirectional=False, bias=True, batch_first=False,
                     seq_len=3, batch_size=5, hidden_size=7, input_size=11, backward=False):
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
            device="cpu"
        )
        bidirectional_mul = 2 if bidirectional else 1

        if batch_first:
            input = torch.randn(batch_size, seq_len, input_size, device="cpu", dtype=dtype, requires_grad=backward)
            hx = torch.randn(num_layers * bidirectional_mul, batch_size, hidden_size, device="cpu", dtype=dtype,
                             requires_grad=backward)
            cx = torch.randn(num_layers * bidirectional_mul, batch_size, hidden_size, device="cpu", dtype=dtype,
                             requires_grad=backward)
        else:
            input = torch.randn(seq_len, batch_size, input_size, device="cpu", dtype=dtype, requires_grad=backward)
            hx = torch.randn(num_layers * bidirectional_mul, batch_size, hidden_size, device="cpu", dtype=dtype,
                             requires_grad=backward)
            cx = torch.randn(num_layers * bidirectional_mul, batch_size, hidden_size, device="cpu", dtype=dtype,
                             requires_grad=backward)

        cpu_output, (cpu_hn, cpu_cn) = rnn(input, (hx, cx))

        rnn = rnn.to(device)
        input = input.to(device)
        hx = hx.to(device)
        cx = cx.to(device)
        output, (hn, cn) = rnn(input, (hx, cx))

        self.assertEqual(cpu_output, output)
        self.assertEqual(cpu_hn, hn)
        self.assertEqual(cpu_cn, cn)

        def get_backward_results(rnn, device, inp, hx, cx, output_grad_presented=True, states_grad_presented=True):
            rnn = rnn.to(device)
            inp, hx, cx = inp.to(device), hx.to(device), cx.to(device)

            output, (hx_out, cx_out) = rnn(inp, (hx, cx))
            assert output_grad_presented or states_grad_presented, "At least some outputs must be used"

            f = 0
            if output_grad_presented:
                f = f + 3 * output.sum()
            if states_grad_presented:
                f = f + (hx_out * cx_out).sum()

            param_names, params = zip(*rnn.named_parameters())
            param_grads = zip(param_names, torch.autograd.grad(f, params, retain_graph=True))

            input_grad, hx_grad, cx_grad = torch.autograd.grad(f, [inp, hx, cx])
            return output, param_grads, input_grad, hx_grad, cx_grad

        if backward:
            grad_cases = [
                dict(output_grad_presented=True, states_grad_presented=True),
                dict(output_grad_presented=False, states_grad_presented=True),
                dict(output_grad_presented=True, states_grad_presented=False),
            ]

            for grad_case in grad_cases:
                cpu_output, cpu_weights_grad, cpu_input_grad, cpu_hx_grad, cpu_cx_grad =\
                    get_backward_results(rnn, "cpu", input, hx, cx, **grad_case)
                mps_output, mps_weights_grad, mps_input_grad, mps_hx_grad, mps_cx_grad =\
                    get_backward_results(rnn, device, input, hx, cx, **grad_case)

                self.assertEqual(cpu_hx_grad, mps_hx_grad)
                self.assertEqual(cpu_cx_grad, mps_cx_grad)
                self.assertEqual(cpu_output, mps_output)
                self.assertEqual(cpu_input_grad, mps_input_grad)
                for (cpu_name, cpu_weight_grad), (mps_name, mps_weight_grad) in zip(cpu_weights_grad, mps_weights_grad):
                    self.assertEqual(cpu_weight_grad, mps_weight_grad,
                                     f"mismatch in cpu:{cpu_name} vs mps:{mps_name}, layers: {num_layers}")

    LSTM_TEST_CASES = [
        dict(),  # default
        dict(batch_first=True),
        dict(bias=False),
        dict(bidirectional=True),
        dict(batch_first=True, bias=False),
        dict(bidirectional=True, bias=False),
        dict(bidirectional=True, batch_first=True),
        dict(bidirectional=True, batch_first=True, bias=False)
    ]

    def test_lstm_forward(self, device="mps", dtype=torch.float32):
        for num_layers in [1, 2, 5]:
            for test_options in self.LSTM_TEST_CASES:
                self._lstm_helper(num_layers=num_layers, dtype=dtype, device=device, **test_options)

    def test_lstm_backward(self, device="mps", dtype=torch.float32):
        for num_layers in [1, 2, 5]:
            for test_options in self.LSTM_TEST_CASES:
                self._lstm_helper(num_layers=num_layers, dtype=dtype, device=device, backward=True, **test_options)

    def test_RNN_cell_no_broadcasting(self):
        def test(cell_module, input, hx, input_size, hidden_size):
            cell = cell_module(input_size, hidden_size, device='mps')
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            test(nn.RNNCell, input, bad_hx, input_size, hidden_size)
            test(nn.GRUCell, input, bad_hx, input_size, hidden_size)
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        hidden_size = 20
        input_size = 10
        input = torch.randn(3, input_size, device='mps')
        bad_hx = torch.randn(1, hidden_size, device='mps')
        good_hx = torch.randn(3, hidden_size, device='mps')

        # Test hidden/input batch size broadcasting
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test hx's hidden_size vs module's hidden_size broadcasting
        bad_hx = torch.randn(3, 1)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test input's input_size vs module's input_size broadcasting
        bad_input = torch.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    def test_LSTM_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = torch.randn(3, 10, device='mps')
            hx = torch.randn(3, 20, device='mps')
            cx = torch.randn(3, 20, device='mps')
            lstm = nn.LSTMCell(10, 20, bias=bias, device='mps')
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx + cx).sum().backward()

    def test_LSTM_cell_forward_input_size(self):
        input = torch.randn(3, 11, device='mps')
        hx = torch.randn(3, 20, device='mps')
        cx = torch.randn(3, 20, device='mps')
        lstm = nn.LSTMCell(10, 20, device='mps')
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))

    def test_LSTM_cell_forward_hidden_size(self):
        input = torch.randn(3, 10, device='mps')
        hx = torch.randn(3, 21, device='mps')
        cx = torch.randn(3, 20, device='mps')
        lstm = nn.LSTMCell(10, 20, device='mps')
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))


class TestFallbackWarning(TestCase):
    # TODO: Remove once test_testing.py is running on MPS devices
    def test_no_warning_on_import(self):
        out = subprocess.check_output(
            [sys.executable, "-W", "all", "-c", "import torch"],
            stderr=subprocess.STDOUT,
            # On Windows, opening the subprocess with the default CWD makes `import torch`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),).decode("utf-8")
        self.assertEqual(out, "")

    def _get_not_implemented_op(self):
        # This can be changed once we actually implement 'lcm'
        # Should return fn, args, kwargs, string_version
        return (torch.lcm,
                [torch.tensor([1], device='mps'), torch.tensor([2], device='mps')], {},
                "torch.lcm(torch.tensor([1], device='mps'), torch.tensor([2], device='mps'))")

    def test_error_on_not_implemented(self):
        fn, args, kwargs, _ = self._get_not_implemented_op()

        with self.assertRaisesRegex(NotImplementedError, "not currently implemented for the MPS device"):
            fn(*args, **kwargs)

    def test_warn_on_not_implemented_with_fallback(self):
        _, _, _, op = self._get_not_implemented_op()
        script = f"""
import os
# MUST happen before pytorch's import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings

with warnings.catch_warnings(record=True) as w:
    import torch

if len(w) > 0:
    print(w)
    exit(1)

# This should run just fine and raise warning about perf
with warnings.catch_warnings(record=True) as w:
    {op}

if len(w) != 1:
    print(w)
    exit(2)

"""
        try:
            subprocess.check_output(
                [sys.executable, '-W', 'all', '-c', script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(False, "There was a warning when importing torch when PYTORCH_ENABLE_MPS_FALLBACK is set." +
                                       e.output.decode("utf-8"))
            elif e.returncode == 2:
                self.assertTrue(False, "There wasn't exactly one warning when running not implemented op with "
                                f"PYTORCH_ENABLE_MPS_FALLBACK set. {e.output}")
            else:
                self.assertTrue(False, "Running a not implemented op failed even though PYTORCH_ENABLE_MPS_FALLBACK is set. " +
                                       e.output.decode("utf-8"))

class TestNoRegression(TestCase):
    def test_assert_close(self):
        a = torch.ones(1, device="mps")
        b = torch.zeros(1, device="mps")
        inf = a / b
        nan = b / b

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            torch.testing.assert_close(a, inf)

        # TODO: The NaN test is failing when all the tests in test_mps are run
        # together but passes when run separately. There seems to be memory
        # corruption which needs to be fixed for this test to be enabled.
        # with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            # torch.testing.assert_close(a, nan)

    def test_double_error(self):
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = torch.ones(2, dtype=torch.float64, device="mps")

        a = torch.ones(2, device="mps")
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = a.double()

    def test_legacy_constructor(self):
        a = torch.ones(2, device="mps")

        b = a.new(1)

    def test_serialization_map_location(self):

        # Ensures that cpu Tensor can be loaded on mps
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2)
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f, map_location="mps")

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "mps")

        # Ensures that mps Tensors can be loaded on mps
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2, device="mps")
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f)

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "mps")

        # Ensures that mps Tensors can be loaded on cpu
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2, device="mps")
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f, map_location="cpu")

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "cpu")

        # Ensures that `mps:0` Tensors can be loaded on mps
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2, device="mps:0")
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f, map_location="mps:0")

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "mps")


MPS_DTYPES = get_all_dtypes()
for t in [torch.double, torch.cdouble, torch.cfloat, torch.bfloat16]:
    del MPS_DTYPES[MPS_DTYPES.index(t)]

MPS_GRAD_DTYPES = [torch.float32, torch.float16]


class TestConsistency(TestCaseMPS):
    # TODO: This is only used while some ops are being added.
    # This list should contain all ops and dtypes eventually
    # This can be generated automatically in the `new_mps_allowlist.txt` file
    # by doing `EXPECTTEST_ACCEPT=1 python test_mps.py TestConsistencyCPU`
    # You most likely do NOT want to modify this manually

    FP16_LOW_PRECISION_LIST = {
        'add', 'sub', 'div',
        '__rdiv__', '__rmul__',
        'nn.functional.huber_loss',
        'true_divide', 'kron',
        'gradient', 'var', 'std', 'ldexp',
        'linalg.vector_norm', 'lerp',
        'addr', 'var_mean',
        'var_mean_unbiased',
        'acosh', 'asinh', 'asin',
        'masked.std',
        'nn.functional.normalize',
        'nn.functional.triplet_margin_loss',
        'nn.functional.triplet_margin_with_distance_loss',
        'nn.functional.batch_norm',
        'nn.functional.instance_norm',
        'round', 'xlogy', 'addcmul',
        'nn.functional.max_pool2d',
        'nn.functional.gelu',
        'nn.functional.glu',
        '_native_batch_norm_legit',
        'native_batch_norm',

        # for macOS 12
        'masked.normalize', 'masked.sum', 'masked.var',
        'outer',
        'sum_to_size', 'sum',
        'mul',
        'nansum', 'nanmean',
        'norm',
    }

    FP32_LOW_PRECISION_LIST = {
        # conv2d and conv_transpose2d results have a very small
        # difference compared to CPU/CUDA, so we use lower precision on FP32
        'nn.functional.conv2d',
        'nn.functional.conv_transpose2d',
        'matmul', '__rmatmul__',
        'linalg.multi_dot',
        'addbmm',
    }

    # Used for accept mode only
    NEW_ALLOW_LIST = defaultdict(list)
    NEW_ALLOW_LIST_GRAD = defaultdict(list)

    @ops(mps_ops_modifier(test_consistency_op_db), allowed_dtypes=MPS_DTYPES + [torch.complex64])
    def test_output_match(self, device, dtype, op):
        self.assertEqual(device, "cpu")

        def get_samples():
            return op.sample_inputs(device, dtype, requires_grad=(dtype.is_floating_point or dtype.is_complex))
        cpu_samples = get_samples()

        for cpu_sample in cpu_samples:
            #
            # Forward check
            #
            mps_sample = cpu_sample.transform(
                lambda x: x.detach().to("mps").requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x)

            cpu_args = [cpu_sample.input] + list(cpu_sample.args)
            cpu_kwargs = cpu_sample.kwargs
            mps_args = [mps_sample.input] + list(mps_sample.args)
            mps_kwargs = mps_sample.kwargs

            # for tensor_split(), the second tensor arg ("tensor_indices_or_sections") must be on CPU only
            if op.name == "tensor_split" and isinstance(mps_args[1], torch.Tensor):
                mps_args[1] = cpu_args[1]

            cpu_out = op(*cpu_args, **cpu_kwargs)
            mps_out = op(*mps_args, **mps_kwargs)

            if (op.name in self.FP32_LOW_PRECISION_LIST) and dtype == torch.float32:
                atol = 1e-4
                rtol = 3e-5
            elif op.name in self.FP16_LOW_PRECISION_LIST and dtype == torch.float16:
                atol = 1e-2
                rtol = 1e-2
            elif op.name in ['nn.functional.conv_transpose1d',
                             'nn.functional.conv_transpose2d',
                             'nn.functional.conv_transpose3d',
                             '__rmatmul__', 'addbmm', 'addmv',
                             'baddbmm', 'cov', 'matmul', 'mv'] and dtype == torch.float16:
                atol = 5e-2
                rtol = 5e-2
            elif op.name == "masked.mean":
                atol = 7e-4
                rtol = 2e-3
            elif op.name == "native_layer_norm":
                atol = 1e-4
                rtol = 1.3e-5
            elif op.name in ["pow", "__rpow__"]:
                atol = 1e-6
                rtol = 4e-6
            elif op.name == "nn.functional.interpolate":
                atol = 1e-3
                rtol = 1e-4
            elif op.name == "nn.functional.upsample_bilinear" and dtype == torch.uint8:
                atol = 1.0
                rtol = 0.0
            else:
                atol = None
                rtol = None

            self.assertEqual(cpu_out, mps_out, atol=atol, rtol=rtol)


    @ops(mps_ops_grad_modifier(copy.deepcopy(test_consistency_op_db)), allowed_dtypes=MPS_GRAD_DTYPES)
    def test_output_grad_match(self, device, dtype, op):
        self.assertEqual(device, "cpu")

        def get_samples():
            return op.sample_inputs(device, dtype, requires_grad=(dtype.is_floating_point or dtype.is_complex))
        cpu_samples = get_samples()

        for cpu_sample in cpu_samples:
            #
            # Forward check
            #
            forward_failed = False
            mps_sample = cpu_sample.transform(
                lambda x: x.detach().to("mps").requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x)

            cpu_args = [cpu_sample.input] + list(cpu_sample.args)
            cpu_kwargs = cpu_sample.kwargs
            mps_args = [mps_sample.input] + list(mps_sample.args)
            mps_kwargs = mps_sample.kwargs

            # for tensor_split(), the second tensor arg ("tensor_indices_or_sections") must be on CPU only
            if op.name == "tensor_split" and isinstance(mps_args[1], torch.Tensor):
                mps_args[1] = cpu_args[1]

            cpu_out = op(*cpu_args, **cpu_kwargs)
            mps_out = op(*mps_args, **mps_kwargs)

            if op.name in self.FP32_LOW_PRECISION_LIST and dtype == torch.float32:
                atol = 1e-4
                rtol = 3e-5
            elif op.name in self.FP16_LOW_PRECISION_LIST and dtype == torch.float16:
                atol = 1e-2
                rtol = 1e-2
            elif op.name in ['nn.functional.conv_transpose1d',
                             'nn.functional.conv_transpose2d',
                             'nn.functional.conv_transpose3d',
                             '__rmatmul__', 'addbmm', 'addmv',
                             'baddbmm', 'cov', 'matmul', 'mv'] and dtype == torch.float16:
                atol = 5e-2
                rtol = 5e-2
            elif (op.name == "masked.mean"):
                atol = 7e-4
                rtol = 2e-3
            elif (op.name == "native_layer_norm"):
                atol = 1e-4
                rtol = 1.3e-5
            elif op.name in ["renorm", "norm", "linalg.norm"] and dtype == torch.float16:
                atol = 7e-4
                rtol = 1.5e-3
            elif op.name == "unique" and cpu_kwargs["sorted"] is False:
                continue
            elif op.name == "nn.functional.interpolate":
                atol = 1e-3
                rtol = 1e-4
            else:
                atol = None
                rtol = None

            self.assertEqual(cpu_out, mps_out, atol=atol, rtol=rtol)

            #
            # Backward check
            #
            if forward_failed:
                # We would've failed immediately anyway, but this error is clearer
                # We error instead of continuing so that all_backward_pass would not be True
                raise RuntimeError("Forward pass already failed")

            cpu_out = (cpu_out,) if isinstance(cpu_out, torch.Tensor) else tuple(cpu_out)
            mps_out = (mps_out,) if isinstance(mps_out, torch.Tensor) else tuple(mps_out)

            def req_grad(t):
                return isinstance(t, torch.Tensor) and t.requires_grad

            diff_cpu_out = tuple(t for t in cpu_out if req_grad(t))
            diff_mps_out = tuple(t for t in mps_out if req_grad(t))
            diff_cpu_arg = tuple(t for t in pytree.tree_flatten((cpu_args, cpu_kwargs))[0] if req_grad(t))
            diff_mps_arg = tuple(t for t in pytree.tree_flatten((mps_args, mps_kwargs))[0] if req_grad(t))
            self.assertEqual(len(diff_cpu_out), len(diff_mps_out))
            self.assertEqual(len(diff_cpu_arg), len(diff_mps_arg))

            if len(diff_cpu_out) == 0:
                continue
            # rand_like does not work with certain dtypes, so cast to double and cast back
            cpu_grad_outputs = tuple(torch.rand_like(t, dtype=torch.double).to(dtype=t.dtype) for t in diff_cpu_out)
            mps_grad_outputs = tuple(t.to("mps") for t in cpu_grad_outputs)

            # Compare computed gradients with cpu given random grad_output vector
            # Sometimes when the derivative is 0, we just don't bother creating the graph
            # allow_unused is needed in those cases.
            cpu_grad_inputs = torch.autograd.grad(diff_cpu_out, diff_cpu_arg, grad_outputs=cpu_grad_outputs, allow_unused=True)
            mps_grad_inputs = torch.autograd.grad(diff_mps_out, diff_mps_arg, grad_outputs=mps_grad_outputs, allow_unused=True)

            self.assertEqual(cpu_grad_inputs, mps_grad_inputs, atol=atol, rtol=rtol)


class TestErrorInputs(TestCase):
    _ignore_not_implemented_error = True

    @ops(mps_ops_error_inputs_modifier(test_error_inputs_op_db), dtypes=OpDTypes.none)
    def test_error_inputs(self, device, op):
        self.assertEqual(device, "mps:0")

        mps_samples = op.error_inputs(device)

        for mps_sample in mps_samples:
            mps_sample_input = mps_sample.sample_input
            error_type = mps_sample.error_type
            error_regex = mps_sample.error_regex

            mps_args = [mps_sample_input.input] + list(mps_sample_input.args)
            mps_kwargs = mps_sample_input.kwargs

            # for tensor_split(), the second tensor arg ("tensor_indices_or_sections") must be on CPU only
            if (op.name == "tensor_split" and isinstance(mps_args[1], torch.Tensor)):
                mps_args[1] = mps_args[1].cpu()

            with self.assertRaisesRegex(error_type, error_regex):
                op(*mps_args, **mps_kwargs)


# Copied from `TestCommon` in `test_ops.py`, just enough to duplicate the `test_numpy_ref` for MPS
@skipIfSlowGradcheckEnv
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # This is the MPS equivalent of `test_numpy_ref` from `test_ops.py`. It lives over here while
    # MPS still requires some fairly heavy special casing in the test framework.
    # When MPS becomes more consistent, this can probably be merged with that test using
    # `@dtypesIfMPS(torch.float32)`, but for now, the assertions themselves need to be loosened
    @suppress_warnings
    # MPS only supports float32
    @ops(_ref_test_ops, allowed_dtypes=(torch.float32,))
    def test_numpy_ref_mps(self, device, dtype, op):
        # Unlike `test_numpy_ref`, this test compares in `float32` since at the time of this test's creation MPS
        # does not support float64 Tensors.
        # A few ops are currently broken on their reference inputs, but not their sample inputs. These should
        # get patched up and this workaround removed.
        broken_on_ref_inputs = op.name in ['clamp', 'where']
        inputs = op.reference_inputs(device, dtype) if not broken_on_ref_inputs else op.sample_inputs(device, dtype)
        for sample_input in inputs:
            self.compare_with_reference(op, op.ref, sample_input)

    @dtypes(*get_all_dtypes())
    def test_tensor_creation(self, device, dtype):
        def ones(device):
            return torch.ones((2, 2), dtype=dtype, device=device)
        if dtype not in MPS_DTYPES:
            with self.assertRaises(TypeError):
                ones(device)
        else:
            mps_tensor = ones(device)
            cpu_tensor = ones("cpu")
            self.assertEqual(mps_tensor.cpu(), cpu_tensor)

# TODO: Actually instantiate that test for the "mps" device to better reflect what it is doing.
# This requires mps to be properly registered in the device generic test framework which is not the
# case right now. We can probably use `allow_mps` introduced in https://github.com/pytorch/pytorch/pull/87342
# to achieve this.
instantiate_device_type_tests(TestConsistency, globals(), only_for="cpu")
instantiate_device_type_tests(TestErrorInputs, globals(), allow_mps=True, only_for="mps")
instantiate_device_type_tests(TestCommon, globals(), allow_mps=True, only_for="mps")

if __name__ == "__main__":
    run_tests()
