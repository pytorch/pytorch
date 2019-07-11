import io
import tempfile
import unittest
import sys
from itertools import repeat
import os
from contextlib import contextmanager
import threading
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue

import torch
import torch.cuda
import torch.cuda.comm as comm
from torch import multiprocessing as mp
from torch._six import inf, nan

from test_torch import _TestTorchMixin

from common_methods_invocations import tri_tests_args, tri_large_tests_args, \
    _compare_trilu_indices, _compare_large_trilu_indices
from common_utils import TestCase, get_gpu_type, to_gpu, freeze_rng_state, run_tests, \
    PY3, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN, skipIfRocm, TEST_NUMPY, TEST_WITH_ROCM, \
    load_tests, slowTest, skipCUDANonDefaultStreamIf

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA and TEST_MULTIGPU from common_cuda here,
# because if we do that, the TEST_CUDNN line from common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if not TEST_CUDA:
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811

TEST_MAGMA = TEST_CUDA
TEST_LARGE_TENSOR = TEST_CUDA
if TEST_CUDA:
    torch.ones(1).cuda()  # has_magma shows up after cuda is initialized
    TEST_MAGMA = torch.cuda.has_magma
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9

floating_set = {torch.FloatTensor, torch.DoubleTensor, torch.cuda.FloatTensor,
                torch.cuda.DoubleTensor, torch.HalfTensor, torch.cuda.HalfTensor}


def is_floating(t):
    if not isinstance(t, type):
        raise TypeError('t should be an instance of type')
    assert t != torch.autograd.Variable
    return t in floating_set


def is_half(t):
    if isinstance(t, torch.Tensor):
        return t.dtype == torch.float16
    assert isinstance(t, type)
    assert t != torch.autograd.Variable
    return t in [torch.HalfTensor, torch.cuda.HalfTensor]


types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.CharTensor,
    torch.ByteTensor,
    torch.HalfTensor,
]

signed_types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.CharTensor,
]

unsigned_types = [
    torch.ByteTensor,
]

float_types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.HalfTensor,
]

float_types_no_half = [
    torch.FloatTensor,
    torch.DoubleTensor,
]


def number(floating, integer, t):
    return floating if is_floating(t) else integer


def cast_tensor(tensor, t):
    return t(tensor.size()).copy_(tensor)

S = 10
M = 50
G = 275000000


def make_tensor(t, *sizes):
    if 'Half' in t.__name__:
        return t(*sizes).copy_(torch.randn(*sizes))
    else:
        tensor = t(*sizes)
        if tensor.is_floating_point():
            return tensor.normal_()
        else:
            return tensor.random_(0, 10)


def make_sparse_tensor(t, n, *sizes):
    assert t.is_sparse
    tensor = t()
    i = tensor._indices()
    i = i.new(len(sizes), n).copy_(
        torch.cat([torch.LongTensor(1, n).random_(s) for s in sizes], 0))
    v = tensor._values()
    v = v.new(n).copy_(torch.randn(n))
    return t(i, v, torch.Size(sizes))


def tensor_clamp(t, min, max):
    if is_half(t):
        return t.float().clamp(min, max).half()
    else:
        return t.clamp(min, max)


def tensor_mul(t, scale):
    if is_half(t):
        return t.float().mul(scale).half()
    else:
        return t.mul(scale)


def tensor_abs_(t):
    if is_half(t):
        return t.float().abs_().half()
    else:
        return t.abs_()


def constant_tensor_sub(a, b):
    # helper function to address const - torch.HalfTensor where it doesn't
    # have resize_as()
    if is_half(b):
        return (a - b.float()).half()
    else:
        return a - b


def constant_tensor_add(a, b):
    # helper function to address const + torch.HalfTensor where it doesn't
    # have add()
    if is_half(b):
        return (a + b.float()).half()
    else:
        return a + b


def small_0d(t):
    return make_tensor(t, (1,)).squeeze()


def small_2d(t):
    return make_tensor(t, S, S)


def small_2d_scaled(t, scale=10):
    return tensor_mul(make_tensor(t, S, S), scale)


def small_2d_oneish(t):
    if is_floating(t):
        return tensor_clamp(make_tensor(t, S, S), min=0.99, max=1.01)
    else:
        return t(S, S).fill_(1)


def small_3d(t):
    return make_tensor(t, S, S, S)


def medium_1d(t):
    return make_tensor(t, M)


def medium_2d(t):
    return make_tensor(t, M, M)


def medium_2d_expanded(t):
    return t(1).expand(M, M)


def medium_2d_scaled(t, scale=10):
    return tensor_mul(make_tensor(t, M, M), scale)


def small_3d_ones(t):
    return t(S, S, S).copy_(torch.ones(S, S, S))


def small_3d_positive(t):
    # In div_tensor(), half cannot achieve float precision
    min_val = 1e-3 if is_floating(t) and not is_half(t) else 2
    return tensor_clamp(make_tensor(t, S, S, S), min_val, 120)


def small_3d_unique(t):
    return t(S, S, S).copy_(torch.arange(1, S * S * S + 1).view(S, S, S))


def small_1d_lapack(t):
    return t(1, 3).copy_(torch.arange(1, 4).view(3))


def small_2d_lapack(t):
    return t(3, 3).copy_(torch.arange(1, 10).view(3, 3))


def small_2d_lapack_skinny(t):
    return t(3, 4).copy_(torch.arange(1, 13).view(3, 4))


def small_2d_lapack_fat(t):
    return t(4, 3).copy_(torch.arange(1, 13).view(4, 3))


def large_2d_lapack(t):
    return t(1000, 1000).normal_()


def giant_1d_ones(t):
    return t(G).copy_(torch.ones(G))


def long_type(t):
    return torch.cuda.LongTensor if 'cuda' in t.__module__ else torch.LongTensor


def new_t(*sizes):
    def tmp(t):
        return t(*sizes).copy_(torch.randn(*sizes))
    return tmp

# Content of each tuple:
# - function name
# - constructor for the tensor,    signature: fn(tensor_type) -> tensor
# - constructor for the arguments, signature: fn(tensor_type) -> list
# - postfix name for the test (must be unique for a given function) (default='')
# - tensor types to use (default=types)
# - disable inplace test, if set to True, no inplace test will be done (default=False)
# - decorator, e.g., unittest.skipIf (default is no decorator)
tests = [
    ('add', small_3d, lambda t: [number(3.14, 3, t)]),
    ('add', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('add', small_3d, lambda t: [number(0.2, 2, t), small_3d_positive(t)], 'scalar_tensor'),
    ('sub', small_3d, lambda t: [number(3.14, 3, t)]),
    ('sub', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('mul', small_3d, lambda t: [number(3.14, 3, t)]),
    ('mul', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('mul', small_0d, lambda t: [small_0d(torch.IntTensor)], 'scalar', types, True),
    ('div', small_3d, lambda t: [number(3.14, 3, t)]),
    ('div', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('pow', small_3d, lambda t: [number(3.14, 3, t)], None, float_types),
    ('pow', small_3d, lambda t: [number(1., 1, t)], 'pow1'),
    ('pow', small_3d, lambda t: [number(2., 2, t)], 'pow2'),
    ('pow', small_3d, lambda t: [number(3., 3, t)], 'pow3'),
    ('pow', small_3d, lambda t: [number(-1., -1, t)], 'pow-1', float_types),
    # HalfTensor gives bad result at pow-2 with data sampled from torch.randn
    ('pow', small_3d, lambda t: [number(-2., -2, t)], 'pow-2', float_types_no_half, False,
        "skipIfRocm:FloatTensor"),
    ('pow', small_3d, lambda t: [tensor_abs_(small_3d(t))], 'tensor', float_types),
    ('addbmm', small_2d, lambda t: [small_3d(t), small_3d(t)], None, float_types),
    ('addbmm', small_2d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('addbmm', small_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars'),
    ('baddbmm', small_3d, lambda t: [small_3d(t), small_3d(t)],),
    ('baddbmm', small_3d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('baddbmm', small_3d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars'),
    ('bmm', small_3d, lambda t: [small_3d(t)], '', float_types_no_half),
    ('addcdiv', small_2d_lapack, lambda t: [tensor_mul(small_2d_lapack(t), 2), small_2d_lapack(t)]),
    ('addcdiv', small_2d_lapack, lambda t: [number(2.8, 1, t), tensor_mul(small_2d_lapack(t), 2), small_2d_lapack(t)],
        'scalar'),
    ('addcmul', small_3d, lambda t: [small_3d(t), small_3d(t)]),
    ('addcmul', small_3d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('addmm', medium_2d, lambda t: [medium_2d(t), medium_2d(t)]),
    ('addmm', medium_2d, lambda t: [number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'scalar'),
    ('addmm', medium_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'two_scalars'),
    ('addmv', medium_1d, lambda t: [medium_2d(t), medium_1d(t)],),
    ('addmv', medium_1d, lambda t: [number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'scalar'),
    ('addmv', medium_1d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'two_scalars'),
    ('addr', medium_2d, lambda t: [medium_1d(t), medium_1d(t)]),
    ('addr', medium_2d, lambda t: [number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'scalar'),
    ('addr', medium_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'two_scalars'),
    ('atan2', medium_2d, lambda t: [medium_2d(t)], None, float_types + [torch.HalfTensor]),
    ('fmod', small_3d, lambda t: [3], 'value',),
    ('fmod', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('chunk', medium_2d, lambda t: [4],),
    ('chunk', medium_2d, lambda t: [4, 1], 'dim'),
    ('chunk', medium_2d, lambda t: [4, -2], 'neg_dim'),
    ('clamp', medium_2d_scaled, lambda t: [-1, 5], None, signed_types),
    ('clamp', medium_2d_scaled, lambda t: [1, 5], None, unsigned_types),
    ('clone', medium_2d, lambda t: [],),
    ('contiguous', medium_2d, lambda t: [],),
    ('cross', new_t(M, 3, M), lambda t: [new_t(M, 3, M)(t)],),
    ('cumprod', small_3d, lambda t: [1]),
    ('cumprod', small_3d, lambda t: [-1], 'neg_dim'),
    ('cumsum', small_3d, lambda t: [1]),
    ('cumsum', small_3d, lambda t: [-1], 'neg_dim'),
    ('dim', small_3d, lambda t: [],),
    ('dist', small_2d, lambda t: [small_2d(t)]),
    ('dist', small_2d, lambda t: [small_2d(t), 3], '3_norm'),
    ('dist', small_2d, lambda t: [small_2d(t), 2.5], '2_5_norm'),
    ('dot', medium_1d, lambda t: [medium_1d(t)], '', types, False, "skipIfRocm:HalfTensor"),
    ('element_size', medium_1d, lambda t: [],),
    ('eq', small_3d_ones, lambda t: [small_3d(t)],),
    ('eq', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('ne', small_3d_ones, lambda t: [small_3d(t)],),
    ('ne', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('equal', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('equal', small_3d_ones, lambda t: [small_3d(t)],),
    ('expand', new_t(M, 1, M), lambda t: [M, 4, M],),
    ('expand_as', new_t(M, 1, M), lambda t: [new_t(M, 4, M)(t)],),
    ('fill', medium_2d, lambda t: [number(3.14, 3, t)]),
    ('ge', medium_2d, lambda t: [medium_2d(t)],),
    ('le', medium_2d, lambda t: [medium_2d(t)],),
    ('gt', medium_2d, lambda t: [medium_2d(t)],),
    ('lt', medium_2d, lambda t: [medium_2d(t)],),
    ('is_contiguous', medium_2d, lambda t: [],),
    # TODO: can't check negative case - GPU copy will be contiguous
    ('is_same_size', medium_2d, lambda t: [small_3d(t)], 'negative'),
    ('is_same_size', medium_2d, lambda t: [medium_2d(t)], 'positive'),
    ('is_set_to', medium_2d, lambda t: [medium_2d(t)],),
    # TODO: positive case
    ('kthvalue', small_3d_unique, lambda t: [3],),
    ('kthvalue', small_3d_unique, lambda t: [3, 1], 'dim'),
    ('kthvalue', small_3d_unique, lambda t: [3, -1], 'neg_dim'),
    ('lerp', small_3d, lambda t: [small_3d(t), 0.3]),
    ('max', small_3d_unique, lambda t: []),
    ('max', small_3d_unique, lambda t: [1], 'dim'),
    ('max', small_3d_unique, lambda t: [-1], 'neg_dim'),
    ('max', medium_2d, lambda t: [medium_2d(t)], 'elementwise'),
    ('min', small_3d_unique, lambda t: []),
    ('min', small_3d_unique, lambda t: [1], 'dim'),
    ('min', small_3d_unique, lambda t: [-1], 'neg_dim'),
    ('min', medium_2d, lambda t: [medium_2d(t)], 'elementwise'),
    ('mean', small_3d, lambda t: []),
    ('mean', small_3d, lambda t: [-1], 'neg_dim'),
    ('mean', small_3d, lambda t: [1], 'dim'),
    ('mean', giant_1d_ones, lambda t: [], '64bit_indexing',
        # Double here because otherwise the CPU result will be
        # wrong.
        [torch.DoubleTensor]),
    ('mode', small_3d, lambda t: []),
    ('mode', small_3d, lambda t: [1], 'dim'),
    ('mode', small_3d, lambda t: [-1], 'neg_dim'),
    ('mvlgamma', lambda t: tensor_clamp(small_2d(t), 0.1, 10), lambda t: [1], '2d_p=1', float_types_no_half),
    ('mvlgamma', lambda t: tensor_clamp(small_2d(t), 0.6, 10), lambda t: [2], '2d_p=2', float_types_no_half),
    ('remainder', small_3d, lambda t: [3], 'value',),
    ('remainder', small_3d, lambda t: [-3], 'negative_value', signed_types),
    ('remainder', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('remainder', small_3d, lambda t: [constant_tensor_sub(0, small_3d_positive(t))], 'negative_tensor', signed_types),
    ('std', small_3d, lambda t: []),
    ('std', small_3d, lambda t: [1], 'dim', types, False),
    ('std', small_3d, lambda t: [-1], 'neg_dim', types, False),
    ('var', small_3d, lambda t: []),
    ('var', small_3d, lambda t: [1], 'dim'),
    ('var', small_3d, lambda t: [-1], 'neg_dim'),
    ('ndimension', small_3d, lambda t: [],),
    ('nelement', small_3d, lambda t: [],),
    ('numel', small_3d, lambda t: [],),
    ('narrow', small_3d, lambda t: [1, 3, 2],),
    ('narrow', small_3d, lambda t: [-1, 3, 2], 'neg_dim'),
    ('nonzero', small_3d, lambda t: [], '', types, False),
    ('norm', small_3d, lambda t: []),
    ('norm', small_3d, lambda t: [3], '3_norm'),
    ('norm', small_3d, lambda t: [3, 0], '3_norm_dim'),
    ('norm', small_3d, lambda t: [3, -2], '3_norm_neg_dim'),
    ('ones', small_3d, lambda t: [1, 2, 3, 4, 5],),
    ('permute', new_t(1, 2, 3, 4), lambda t: [2, 1, 3, 0],),
    ('put_', new_t(2, 5, 3), lambda t: [long_type(t)([[0], [-2]]), t([[3], [4]])], '', types, False),
    ('put_', new_t(2, 3), lambda t: [long_type(t)([]), t([])], 'empty'),
    ('put_', new_t(2, 2), lambda t: [long_type(t)([[1], [-3]]), t([[1], [2]]), True], 'accumulate'),
    ('prod', small_2d_oneish, lambda t: []),
    ('prod', small_3d, lambda t: [1], 'dim'),
    ('prod', small_3d, lambda t: [-1], 'neg_dim'),
    ('sum', small_2d, lambda t: []),
    ('sum', small_3d, lambda t: [1], 'dim'),
    ('sum', small_3d, lambda t: [-1], 'neg_dim'),
    ('renorm', small_3d, lambda t: [2, 1, 1], '2_norm'),
    ('renorm', small_3d, lambda t: [2, -1, 1], '2_norm_neg_dim'),
    ('renorm', small_3d, lambda t: [1.5, 1, 1], '1_5_norm'),
    ('repeat', small_2d, lambda t: [2, 2, 2],),
    ('size', new_t(1, 2, 3, 4), lambda t: [],),
    ('size', new_t(1, 2, 3, 4), lambda t: [1], 'dim'),
    ('size', new_t(1, 2, 3, 4), lambda t: [-2], 'neg_dim'),
    ('sort', small_3d_unique, lambda t: [], ''),
    ('sort', small_3d_unique, lambda t: [1], 'dim'),
    ('sort', small_3d_unique, lambda t: [-1], 'neg_dim'),
    ('sort', small_3d_unique, lambda t: [1, True], 'dim_descending'),
    ('sort', small_3d_unique, lambda t: [-1, True], 'neg_dim_descending'),
    ('split', small_3d, lambda t: [2],),
    ('split', small_3d, lambda t: [2, 1], 'dim'),
    ('split', small_3d, lambda t: [2, -3], 'neg_dim'),
    ('squeeze', new_t(1, 2, 1, 4), lambda t: [],),
    ('squeeze', new_t(1, 2, 1, 4), lambda t: [2], 'dim'),
    ('squeeze', new_t(1, 2, 1, 4), lambda t: [-2], 'neg_dim'),
    ('t', new_t(1, 2), lambda t: [],),
    ('take', new_t(3, 4), lambda t: [long_type(t)([[0], [-2]])], '', types, False),
    ('transpose', new_t(1, 2, 3, 4), lambda t: [1, 2],),
    ('transpose', new_t(1, 2, 3, 4), lambda t: [-1, -2], 'neg_dim'),
    ('to_list', small_3d, lambda t: [],),
    ('topk', small_3d_unique, lambda t: [2, 1, False, True], 'dim_sort',),
    ('topk', small_3d_unique, lambda t: [2, -1, False, True], 'neg_dim_sort',),
    ('topk', small_3d_unique, lambda t: [2, 1, True, True], 'dim_desc_sort',),
    ('trace', medium_2d, lambda t: []),
    ('tril', medium_2d, lambda t: [],),
    ('tril', medium_2d_expanded, lambda t: [], 'zero_stride', types, True),
    ('tril', medium_2d, lambda t: [2], 'positive'),
    ('tril', medium_2d, lambda t: [-2], 'negative'),
    ('triu', medium_2d, lambda t: [],),
    ('triu', medium_2d_expanded, lambda t: [], 'zero_stride', types, True),
    ('triu', medium_2d, lambda t: [2], 'positive'),
    ('triu', medium_2d, lambda t: [-2], 'negative'),
    ('unsqueeze', new_t(2, 3, 4), lambda t: [2],),
    ('unsqueeze', new_t(2, 3, 4), lambda t: [-2], 'neg_dim'),
    ('view', small_3d, lambda t: [100, 10], 'contiguous'),
    ('view_as', small_3d, lambda t: [make_tensor(t, 100, 10)],),
    ('zero', small_3d, lambda t: [],),
    ('zeros', small_3d, lambda t: [1, 2, 3, 4],),
    ('eye', small_2d, lambda t: [3, 4],),
    ('flip', small_3d, lambda t: [0], 'd0', types, True),
    ('flip', small_3d, lambda t: [0, 1, 2], 'd012', types, True),
    ('flip', small_3d, lambda t: [0, 2], 'd02', types, True),
    ('flip', small_3d, lambda t: [2, 0], 'd20', types, True),
    ('flip', small_3d, lambda t: [-1], 'neg_d', types, True),
    ('rot90', small_2d, lambda t: [1, [0, 1]], 'k1_d01', types, True),
    ('rot90', small_3d, lambda t: [1, [1, 2]], 'k1_d12', types, True),
    ('rot90', small_3d, lambda t: [1, [1, -1]], 'k1_neg_d', types, True),
    ('rot90', small_3d, lambda t: [], 'default', types, True),
    ('rsqrt', lambda t: constant_tensor_add(1, small_3d(t)), lambda t: [], None, float_types),
    ('sinh', lambda t: tensor_clamp(small_3d(t), -1, 1), lambda t: [], None, float_types),
    ('tan', lambda t: tensor_clamp(small_3d(t), -1, 1), lambda t: [], None, float_types),
    ('__lshift__', lambda t: torch.pow(2, cast_tensor(torch.arange(1, 5), t)),
        lambda t: [2], None, signed_types),
    ('__rshift__', lambda t: torch.pow(2, cast_tensor(torch.arange(3, 7), t)),
        lambda t: [2], None, signed_types),
    # lapack tests
    ('qr', small_2d_lapack, lambda t: [], 'square', float_types, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('qr', small_2d_lapack_skinny, lambda t: [], 'skinny', float_types, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('qr', small_2d_lapack_fat, lambda t: [], 'fat', float_types, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('qr', large_2d_lapack, lambda t: [], 'big', float_types, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('geqrf', new_t(20, 20), lambda t: [], None, float_types, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', new_t(10, 10), lambda t: [], 'square', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', lambda t: new_t(10, 10)(t).t(), lambda t: [True], 'square_col_maj',
        float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', new_t(20, 5), lambda t: [True], 'tall_some', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', new_t(20, 5), lambda t: [False], 'tall_all', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', lambda t: new_t(5, 20)(t).t(), lambda t: [True],
        'tall_some_col_maj', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('svd', lambda t: new_t(5, 20)(t).t(), lambda t: [False],
        'tall_all_col_maj', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
    ('eig', new_t(10, 10), lambda t: [True], 'with_eigvec', float_types_no_half, False,
        unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")),
]

# TODO: random functions, cat, gather, scatter, index*, masked*,
#       resize, resizeAs, storage_offset, storage, stride, unfold

custom_precision = {
    'addbmm': 1e-4,
    'addmm': 1e-4,
    'addmv': 1e-4,
    'addr': 1e-4,
    'baddbmm': 1e-4,
    'rsqrt': 1e-4,
    'cumprod': 1e-4,
    'qr': 3e-4,
    'digamma': 1e0,  # large values lead to large absolute error but small relative error
}

custom_half_precision = {
    'add': 1e-2,
    'acos': 1e-3,
    'addbmm': 1e-1,
    'addcdiv': 1e-2,
    'addcmul': 1e-2,
    'addmm': 1e-1,
    'addmv': 1e-2,
    'addr': 1e-2,
    'asin': 1e-3,
    'atan2': 1e-3,
    'atan': 1e-3,
    'baddbmm': 1e-2,
    'cos': 1e-3,
    'cosh': 1e-2,
    'cross': 1e-2,
    'cumprod': 1e-2,
    'cumsum': 1e-2,
    'dist': 1e-2,
    'div': 1e-3,
    'dot': 1e-2,
    'erf': 1e-3,
    'erfc': 1e-3,
    'erfinv': 1e-3,
    'exp': 1e-2,
    'expm1': 1e-2,
    'fill': 1e-3,
    'lerp': 1e-2,
    'lgamma': 1e-2,
    'log': 1e-2,
    'log10': 1e-2,
    'log1p': 1e-3,
    'log2': 1e-2,
    'mean': 1e-3,
    'mul': 1e-2,
    'norm': 1e-1,
    'pow': 1e-1,
    'prod': 1e-3,
    'reciprocal': 1e-1,
    'remainder': 1e-3,
    'renorm': 1e-3,
    'rsqrt': 1e-2,
    'sigmoid': 1e-3,
    'sin': 1e-3,
    'sinh': 1e-3,
    'sqrt': 1e-3,
    'std': 1e-3,
    'sub': 1e-2,
    'sum': 1e-2,
    'tan': 1e-3,
    'tanh': 1e-3,
    'trace': 1e-3,
    'var': 1e-3,
    '__lshift__': 1e-3,
    '__rshift__': 1e-3,
}

simple_pointwise = [
    'abs',
    'sign',
]
for fn in simple_pointwise:
    tests.append((fn, small_3d, lambda t: []))

simple_pointwise_float = [
    'log',
    'log10',
    'log1p',
    'log2',
    'sigmoid',
    'sin',
    'sqrt',
    'tanh',
    'acos',
    'asin',
    'atan',
    'cos',
    'cosh',
    'erf',
    'erfc',
    'erfinv',
    'exp',
    'expm1',
    'reciprocal',
    'floor',
    'frac',
    'neg',
    'round',
    'trunc',
    'ceil',
    'lgamma',
    'digamma',
    'trigamma',
]

for fn in simple_pointwise_float:
    tests.append((fn, small_3d, lambda t: [], None, float_types))

_cycles_per_ms = None


def get_cycles_per_ms():
    """Approximate number of cycles per millisecond for torch.cuda._sleep"""
    global _cycles_per_ms
    if _cycles_per_ms is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        _cycles_per_ms = 1000000 / start.elapsed_time(end)
    return _cycles_per_ms


def compare_cpu_gpu(tensor_constructor, arg_constructor, fn, t, precision=1e-5):
    def tmp(self):
        cpu_tensor = tensor_constructor(t)
        gpu_tensor = to_gpu(cpu_tensor)
        cpu_args = arg_constructor(t)
        gpu_args = [to_gpu(arg) for arg in cpu_args]
        if is_half(t):
            cpu_tensor = cpu_tensor.float()
            cpu_args = [arg.float() if isinstance(arg, torch.Tensor) and is_half(arg) else arg for arg in cpu_args]
        cpu_result = getattr(cpu_tensor, fn)(*cpu_args)
        try:
            gpu_result = getattr(gpu_tensor, fn)(*gpu_args)
        except RuntimeError as e:
            reason = e.args[0]
            data_type_reasons = {'only supports floating-point types',
                                 'unimplemented data type',
                                 'not implemented for'}
            if any(data_type_reason in reason for data_type_reason in data_type_reasons):
                raise unittest.SkipTest('unimplemented data type')
            raise
        except AttributeError as e:
            reason = e.args[0]
            if 'object has no attribute' in reason:
                raise unittest.SkipTest('unimplemented data type')
            raise
        # If one changes, another should change as well
        self.assertEqual(cpu_tensor, gpu_tensor, precision)
        self.assertEqual(cpu_args, gpu_args, precision)
        # Compare results
        if fn == 'element_size' and t.__name__ == 'HalfTensor':
            # Workaround since cpu_result is float
            self.assertEqual(2, gpu_result)
        else:
            self.assertEqual(cpu_result, gpu_result, precision)
    return tmp


class TestCuda(TestCase):
    _do_cuda_memory_leak_check = True
    # See https://github.com/pytorch/pytorch/issues/21589
    # We used to have this turned on for the tests in this file which
    # we had tested to be OK, but when people added new tests to
    # this file, it would trigger nondeterministic failures that
    # are hard to debug.  Since there are KNOWN bugs with our
    # stream handling, we shouldn't turn this on by default.
    # If you decide to make this True, be sure to run the test suite
    # under cuda-memcheck
    _do_cuda_non_default_stream = False
    FIFTY_MIL_CYCLES = 50000000

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch.cuda.current_device()

        m0 = torch.cuda.memory_allocated(device)
        last_m_arr = [torch.cuda.memory_allocated(device)]
        max_m_arr = [torch.cuda.max_memory_allocated(device)]
        last_c_arr = [torch.cuda.memory_cached(device)]
        max_c_arr = [torch.cuda.max_memory_cached(device)]

        def alloc(*size):
            with torch.cuda.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch.cuda.FloatTensor(*size)

        def assert_change(comp=1, empty_cache=False, reset_max_alloc=False, reset_max_cached=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch.cuda.memory_allocated(device)
            new_max_m = torch.cuda.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_c = torch.cuda.memory_cached(device)
            new_max_c = torch.cuda.max_memory_cached(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_c, new_max_c)
            self.assertGreaterEqual(new_max_c, max_c_arr[0])
            last_c_arr[0] = new_c
            max_c_arr[0] = new_max_c

            if empty_cache:
                torch.cuda.empty_cache()
                new_c = torch.cuda.memory_cached(device)
                new_max_c = torch.cuda.max_memory_cached(device)
                self.assertLessEqual(new_c, last_c_arr[0])
                self.assertLessEqual(new_c, new_max_c)
                self.assertEqual(new_max_c, max_c_arr[0])
                last_c_arr[0] = new_c

            if reset_max_alloc:
                torch.cuda.reset_max_memory_allocated(device)
                self.assertEqual(torch.cuda.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.cuda.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch.cuda.memory_cached(device), last_c_arr[0])
                self.assertEqual(torch.cuda.max_memory_cached(device), max_c_arr[0])

            if reset_max_cached:
                torch.cuda.reset_max_memory_cached(device)
                self.assertEqual(torch.cuda.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.cuda.max_memory_allocated(device), max_m_arr[0])
                self.assertEqual(torch.cuda.memory_cached(device), last_c_arr[0])
                self.assertEqual(torch.cuda.max_memory_cached(device), last_c_arr[0])
                max_c_arr[0] = last_c_arr[0]

        assert_change(0)
        assert_change(0, reset_max_alloc=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_max_cached=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch.cuda.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_max_alloc=(i % 2 == 0), reset_max_cached=(i % 2 == 1))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        assert_change(0, reset_max_alloc=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_max_cached=True)
        assert_change(0)
        self.assertEqual(torch.cuda.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_max_alloc=True)
        self.assertEqual(torch.cuda.memory_allocated(device), m0)

        # test empty_cache and reset_max_memory_*
        assert_change(0, empty_cache=True)
        assert_change(0, reset_max_cached=True)
        assert_change(0, reset_max_alloc=True)

    def test_memory_stats(self):
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            pass

    def test_cuda_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    def test_cuda_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_memory_stats_multigpu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device='cuda:0', N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device='cuda')

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 80.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 80, dtype=torch.int8, device='cuda')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_autogpu(self):
        x = torch.randn(5, 5).cuda()
        y = torch.randn(5, 5).cuda()
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(x.get_device(), 0)
        with torch.cuda.device(1):
            z = torch.randn(5, 5).cuda()
            self.assertEqual(z.get_device(), 1)
            q = x.add(y)
            self.assertEqual(q.get_device(), 0)
            w = torch.randn(5, 5).cuda()
            self.assertEqual(w.get_device(), 1)
            self.assertEqual(y.cuda().get_device(), 1)
        z = z.cuda()
        self.assertEqual(z.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_new(self):
        x = torch.randn(3, 3).cuda()
        self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
        self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

        with torch.cuda.device(1):
            self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
            self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_device(self):
        x = torch.randn(5, 5).cuda()
        with torch.cuda.device(1):
            y = x.cuda()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.cuda(), y)
            z = y.cuda(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.cuda(0), z)

        x = torch.randn(5, 5)
        with torch.cuda.device(1):
            y = x.cuda()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.cuda(), y)
            z = y.cuda(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.cuda(0), z)

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch.cuda.Stream(device=x.device)
        s1 = torch.cuda.Stream(device=y.device)
        s2 = torch.cuda.Stream(device=x.device)
        s3 = torch.cuda.Stream(device=y.device)

        # same dst stream different src streams
        with torch.cuda.stream(s0):
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s1):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s2), torch.cuda.stream(s1):
            y.copy_(x)

        s1.synchronize()
        # The copy() is synchronized on the current streams of both src and dst.
        # In the above test, the _sleep() op on s0 will not block the copy() on
        # s2, but both copies are synchronized on s1 in the dst device. Hence,
        # x is copied to y after x_plus_one is copied to y. If x and y are on
        # the same device, both copy() ops are synchronized on s1.
        self.assertEqual(y, x)

        # same src stream different dst streams
        with torch.cuda.stream(s1):
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s0):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s3), torch.cuda.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_streams(self):
        d0 = torch.device('cuda:0')
        x0 = torch.zeros(5, 5, device=d0)

        d1 = torch.device('cuda:1')
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)

        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            self.assertFalse(event.query())
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_copy_broadcast(self):
        x = torch.randn(10, 5)
        y = torch.randn(5, device='cuda')
        x.copy_(y)
        self.assertEqual(x[3], y.cpu())

        x = torch.randn(10, 5, device='cuda')
        y = torch.randn(5)
        x.copy_(y)
        self.assertEqual(x[3].cpu(), y)

    def test_copy_noncontig(self):
        def do_test(d0, d1):
            x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device=d0)
            y = torch.tensor([0, 0, 0, 0, 0, 0], device=d1)
            self.assertNotEqual(x.dtype, y.dtype)

            y[::2].copy_(x[::2])
            self.assertEqual(y, [1, 0, 3, 0, 5, 0])

        do_test('cpu', 'cuda')
        do_test('cuda', 'cpu')
        if TEST_MULTIGPU:
            do_test('cuda:0', 'cuda:1')

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, 0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], 0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.DoubleTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.DoubleTensor))
        self.assertTrue(isinstance(q_copy[3], torch.cuda.IntStorage))
        q_copy[1].fill_(10)
        self.assertTrue(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.cuda(), torch.cuda.DoubleTensor)
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.cuda(), torch.cuda.DoubleStorage)
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    def test_mul_intertype_scalar(self):
        def test_mul(dtype):
            x = torch.tensor(1.5, dtype=dtype, device='cuda')
            y = torch.tensor(3, dtype=torch.int32, device='cuda')

            self.assertEqual(x * y, 4.5)
            self.assertEqual(y * x, 4.5)
            with self.assertRaisesRegex(RuntimeError, "doesn't match the desired"):
                y *= x
            x *= y
            self.assertEqual(x, 4.5)

        test_mul(torch.float16)
        test_mul(torch.float32)
        test_mul(torch.float64)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_type_conversions_same_gpu(self):
        x = torch.randn(5, 5).cuda(1)
        self.assertEqual(x.int().get_device(), 1)
        self.assertEqual(x.type(torch.int).get_device(), 1)
        self.assertEqual(x.to(torch.int).get_device(), 1)

    def test_neg(self):
        _TestTorchMixin._test_neg(self, lambda t: t.cuda())

    def test_bitwise_not(self):
        _TestTorchMixin._test_bitwise_not(self, 'cuda')

    def test_isinf(self):
        _TestTorchMixin._test_isinf(self, lambda t: t.cuda())

    def test_inplace_unary_mem_overlap(self):
        _TestTorchMixin._test_inplace_unary_mem_overlap(self, device='cuda')

    def test_inplace_binary_mem_overlap(self):
        _TestTorchMixin._test_inplace_binary_mem_overlap(self, device='cuda')

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_arithmetic_large_tensor(self):
        x = torch.empty(2**30, device='cuda')

        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        x += 1
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    def _test_broadcast(self, input):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        result = comm.broadcast(input, (0, 1))
        for i, t in enumerate(result):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)
            if input.is_cuda and input.get_device() == i:
                self.assertEqual(t.data_ptr(), input.data_ptr())

    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).cuda())

    def test_min_max_nan(self):
        tests = [(lambda x: x.min(), 'min'),
                 (lambda x: x.max(), 'max'),
                 (lambda x: x.min(0)[0], 'min_dim'),
                 (lambda x: x.max(0)[0], 'max_dim')]
        for f, name in tests:
            a = torch.arange(25.0).view(5, 5)
            a[2, 2] = nan
            actual = f(a.cuda()).cpu()
            expected = f(a).cpu()
            self.assertEqual(torch.isnan(actual), torch.isnan(expected), 'nans for {}'.format(name))
            self.assertEqual(actual[~torch.isnan(actual)],
                             expected[~torch.isnan(expected)], 'nans for {}'.format(name))

    @staticmethod
    def _test_broadcast_coalesced(self, tensors, buffer_size):
        b_tensors = [comm.broadcast(t, (0, 1)) for t in tensors]
        for (_, bt), t in zip(b_tensors, tensors):
            self.assertEqual(bt.get_device(), 1)
            self.assertEqual(bt, t)
            self.assertIsInstance(bt, type(t))

        bc_tensors = comm.broadcast_coalesced(tensors, (0, 1), buffer_size=buffer_size)
        bc_tensors_t = list(zip(*bc_tensors))
        self.assertEqual(b_tensors, bc_tensors_t)
        for (_, bt), (_, bct) in zip(b_tensors, bc_tensors_t):
            self.assertEqual(bt.get_device(), bct.get_device())
            self.assertIsInstance(bct, type(bt))

        # check that tensors on device[0] are returned as-is
        for out_tensors in (b_tensors, bc_tensors_t):
            for inp_t, (out_t, _) in zip(tensors, out_tensors):
                self.assertIs(inp_t, out_t)

        # check that the tensors not on device[0] have different version counters
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for _, t in bc_tensors_t]
        for old_version, (_, t) in zip(versions, bc_tensors_t):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # Note: fails sometimes on the CI, passes on dual gfx906
    def test_broadcast_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 1, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 10, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 5, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 7, 3, 3),
            make_sparse_tensor(torch.cuda.sparse.FloatTensor, 2, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 3, 2, 7),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(self, tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(self, tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        x_cuda = x.cuda(0)
        y_cuda = y.cuda(1)
        result = comm.reduce_add((x_cuda, y_cuda))
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.cpu(), x + y)

    @staticmethod
    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        dup_tensors = [tensors, list(map(lambda t: t.cuda(1), tensors))]

        r_tensors = list(map(comm.reduce_add, zip(*dup_tensors)))
        for r, t in zip(r_tensors, tensors):
            self.assertEqual(r.get_device(), t.get_device())
            self.assertEqual(r, t * 2)
            self.assertEqual(r.type(), t.type())

        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)
        self.assertEqual(r_tensors, rc_tensors)
        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqual(rc.get_device(), r.get_device())
            self.assertEqual(rc.type(), r.type())

        # Since we have both cuda:0 and cuda:1 inputs, the outputs must be new.
        # We can check that they have different version counters.
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for t in rc_tensors]
        for old_version, t in zip(versions, rc_tensors):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 1, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 10, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 5, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 7, 3, 3),
            make_sparse_tensor(torch.cuda.sparse.FloatTensor, 2, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 3, 2, 7),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_reduce_add_coalesced(self, tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_reduce_add_coalesced(self, tensors, num_bytes * 5 // 2)

    def _test_scatter(self, input, chunk_sizes=None, dim=0):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        result = comm.scatter(input, (0, 1), chunk_sizes, dim)
        self.assertEqual(len(result), 2)
        if chunk_sizes is None:
            chunk_sizes = tuple(repeat(input.size(dim) // 2, 2))
        chunk_start = 0
        for i, r in enumerate(result):
            chunk_end = chunk_start + chunk_sizes[i]
            index = [slice(None, None), slice(None, None)]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input[tuple(index)], 0)
            chunk_start = chunk_end

    def test_scatter_cpu(self):
        self._test_scatter(torch.randn(4, 4), dim=0)

    def test_scatter_cpu_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=1)

    def test_scatter_cpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=-2)

    def test_scatter_cpu_sizes(self):
        self._test_scatter(torch.randn(6, 4), chunk_sizes=(2, 4))

    def test_scatter_gpu(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=0)

    def test_scatter_gpu_dim(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=1)

    def test_scatter_gpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=-2)

    def test_scatter_gpu_sizes(self):
        self._test_scatter(torch.randn(6, 4).cuda(), chunk_sizes=(2, 4))

    def _test_gather(self, dim):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        x = torch.randn(2, 5).cuda(0)
        y = torch.randn(2, 5).cuda(1)
        result = comm.gather((x, y), dim)

        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = torch.Size(expected_size)
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.size(), expected_size)

        index = [slice(None, None), slice(None, None)]
        index[dim] = slice(0, x.size(dim))
        self.assertEqual(result[tuple(index)], x)
        index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
        self.assertEqual(result[tuple(index)], y)

        # Bool test case
        t = torch.tensor([[False, True], [True, True]], device='cuda')
        self.assertEqual(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device='cuda')), 
                         torch.tensor([[False, False], [True, True]], device='cuda'))

    def test_gather(self):
        self._test_gather(0)

    def test_gather_dim(self):
        self._test_gather(1)

    def test_from_sequence(self):
        seq = [list(range(i * 4, i * 4 + 4)) for i in range(5)]
        reference = torch.arange(0, 20).resize_(5, 4)
        for t in types:
            cuda_type = get_gpu_type(t)
            self.assertEqual(cuda_type(seq), reference)

    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_cat_autogpu(self):
        x = torch.randn(4, 4).cuda(1)
        y = torch.randn(4, 4).cuda(1)
        z = torch.cat([x, y], 0)
        self.assertEqual(z.get_device(), x.get_device())

    def test_clamp(self):
        _TestTorchMixin._test_clamp(self, 'cuda')

    def test_cat(self):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE).transpose(0, pos_dim).cuda()
            y = torch.rand(17, SIZE, SIZE).transpose(0, pos_dim).cuda()
            z = torch.rand(19, SIZE, SIZE).transpose(0, pos_dim).cuda()

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, 0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, 0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, 0)

        x = torch.randn(20, SIZE, SIZE).cuda()
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE).cuda()
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    def test_cat_empty_legacy(self):
        _TestTorchMixin._test_cat_empty_legacy(self, use_cuda=True)

    def test_cat_empty(self):
        _TestTorchMixin._test_cat_empty(self, use_cuda=True)

    def test_bernoulli(self):
        _TestTorchMixin._test_bernoulli(self, torch.float32, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float32, torch.float16, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float16, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float16, torch.float16, 'cuda')
        # test that it works with integral tensors
        _TestTorchMixin._test_bernoulli(self, torch.uint8, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.uint8, torch.float16, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.int64, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.int64, torch.float16, 'cuda')

    def test_cat_bad_input_sizes(self):
        x = torch.randn(2, 1).cuda()
        y = torch.randn(2, 1, 1).cuda()
        z = torch.randn(2, 1, 1).cuda()
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z]))

        x = torch.randn(2, 1, 2).cuda()
        y = torch.randn(2, 1, 1).cuda()
        z = torch.randn(2, 2, 1).cuda()
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z], dim=1))

    @unittest.skipIf(torch.cuda.device_count() >= 10, "Loading a cuda:9 tensor")
    @unittest.skipIf(not PY3, "Tensor was serialized with Python 3")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:9' restore location
        tensor = torch.randn(2, device='cuda')
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # NB: this might not work in the future if serialization changes
        buf = io.BytesIO(buf.getvalue().replace(b'cuda:0', b'cuda:9'))

        msg = r'Attempting to deserialize object on CUDA device 9'
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    def test_serialization(self):
        x = torch.randn(4, 4).cuda()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        self.assertEqual(x_copy, x)
        self.assertIs(type(x_copy), type(x))
        self.assertEqual(x_copy.get_device(), x.get_device())

    def test_serialization_array_with_empty(self):
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization_remap(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]

        def gpu_remap(storage, location):
            if location == 'cuda:1':
                return storage.cuda(0)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location=gpu_remap)

        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization_remap_dict(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location={'cuda:1': 'cuda:0'})
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_storage_clone(self):
        x = torch.randn(4, 4, device='cuda:1').storage()
        y = x.clone()
        self.assertEqual(x.get_device(), y.get_device())
        for t in ['byte', 'char', 'short', 'int', 'long', 'half', 'double']:
            self.assertEqual(getattr(x, t)().get_device(), x.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_cuda_set_device(self):
        x = torch.randn(5, 5)
        with torch.cuda.device(1):
            self.assertEqual(x.cuda().get_device(), 1)
            torch.cuda.set_device(0)
            self.assertEqual(x.cuda().get_device(), 0)
            with torch.cuda.device(1):
                self.assertEqual(x.cuda().get_device(), 1)
            self.assertEqual(x.cuda().get_device(), 0)
            torch.cuda.set_device(1)
        self.assertEqual(x.cuda().get_device(), 0)

    def test_is_tensor(self):
        for t in types:
            tensor = get_gpu_type(t)()
            self.assertTrue(torch.is_tensor(tensor))
        self.assertTrue(torch.is_tensor(torch.cuda.HalfTensor()))

    def test_cuda_synchronize(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize('cuda')
        torch.cuda.synchronize('cuda:0')
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(torch.device('cuda:0'))

        if TEST_MULTIGPU:
            torch.cuda.synchronize('cuda:1')
            torch.cuda.synchronize(1)
            torch.cuda.synchronize(torch.device('cuda:1'))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize("cpu")

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_current_stream(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.current_stream(device=1)
        s2 = torch.cuda.current_stream(device=0)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s2)

        with torch.cuda.device(d1):
            s0 = torch.cuda.current_stream()
            s1 = torch.cuda.current_stream(1)
            s2 = torch.cuda.current_stream(d0)

        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s1)

        with self.assertRaisesRegex(ValueError,
                                    "Expected a cuda device, but got: cpu"):
            torch.cuda.current_stream(torch.device('cpu'))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.default_stream()

        with torch.cuda.device(d1):
            s1 = torch.cuda.default_stream()

        s2 = torch.cuda.default_stream(device=0)
        s3 = torch.cuda.default_stream(d1)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch.cuda.device(d0):
            self.assertEqual(torch.cuda.current_stream(), s0)

        with torch.cuda.device(d1):
            self.assertEqual(torch.cuda.current_stream(), s1)

        with self.assertRaisesRegex(ValueError,
                                    "Expected a cuda device, but got: cpu"):
            torch.cuda.default_stream(torch.device('cpu'))

    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        # copy 10 MB tensor from CPU-GPU which should take some time
        tensor1 = torch.ByteTensor(10000000).pin_memory()
        tensor2 = tensor1.cuda(non_blocking=True)
        self.assertFalse(default_stream.query())
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_device(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')
        e0 = torch.cuda.Event()

        self.assertEqual(None, e0.device)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device('cuda:0'))
        self.assertEqual(e0.device, torch.device('cuda:0'))
        self.assertEqual(s1.device, torch.device('cuda:1'))
        self.assertEqual(e1.device, torch.device('cuda:1'))

    def test_stream_event_repr(self):
        s = torch.cuda.current_stream()
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.cuda.Event" in e.__repr__())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # Note: fails sometimes on the CI, passes on dual gfx906
    @skipIfRocm
    def test_stream_context(self):
        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.Stream(device=1)
        s2 = torch.cuda.Stream(device=0)

        with torch.cuda.device(s1.device):
            prev_stream_on_cuda1 = torch.cuda.current_stream()

        self.assertEqual(torch.cuda.current_stream(), s0)
        self.assertEqual(0, torch.cuda.current_device())
        with torch.cuda.stream(s1):
            self.assertEqual(torch.cuda.current_stream(), s1)
            self.assertEqual(1, torch.cuda.current_device())
            with torch.cuda.stream(s2):
                self.assertEqual(torch.cuda.current_stream(), s2)
                self.assertEqual(0, torch.cuda.current_device())
                with torch.cuda.stream(s0):
                    self.assertEqual(torch.cuda.current_stream(), s0)
                    self.assertEqual(0, torch.cuda.current_device())
                self.assertEqual(torch.cuda.current_stream(), s2)
                self.assertEqual(0, torch.cuda.current_device())
            self.assertEqual(torch.cuda.current_stream(), s1)
            self.assertEqual(1, torch.cuda.current_device())

        with torch.cuda.device(s1.device):
            self.assertEqual(prev_stream_on_cuda1, torch.cuda.current_stream())

        self.assertEqual(torch.cuda.current_stream(), s0)
        self.assertEqual(0, torch.cuda.current_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu(self):
        default_stream = torch.cuda.current_stream()
        self.assertEqual(default_stream.device, torch.device('cuda:0'))
        stream = torch.cuda.Stream(device=1)
        self.assertEqual(stream.device, torch.device('cuda:1'))
        with torch.cuda.device(1):
            self.assertEqual(
                torch.cuda.current_stream().device, torch.device('cuda:1'))
            self.assertNotEqual(torch.cuda.current_stream(), default_stream)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu_query(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)

        self.assertTrue(s0.query())
        self.assertFalse(s1.query())

        with torch.cuda.device(d0):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        with torch.cuda.device(d1):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        # deliberately using a different device
        with torch.cuda.device(d0):
            s1.synchronize()

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

        with torch.cuda.device(d0):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

        with torch.cuda.device(d1):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu_eq(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            s1 = torch.cuda.current_stream()

        with torch.cuda.device(d1):
            s2 = torch.cuda.current_stream()
            s3 = torch.cuda.current_stream()

        self.assertTrue(s0 == s0)
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)

        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.cuda_stream, s1.cuda_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.cuda_stream, s3.cuda_stream)
        self.assertNotEqual(s0.device, s3.device)

        self.assertEqual(hash(s0), hash(s1))
        self.assertEqual(hash(s2), hash(s3))
        self.assertNotEqual(hash(s0), hash(s3))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @skipIfRocm
    def test_streams_priority(self):
        low, high = torch.cuda.Stream.priority_range()
        s0 = torch.cuda.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device('cuda:0'), s0.device)

        s1 = torch.cuda.Stream(device=1, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device('cuda:1'), s1.device)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_tensor_device(self):
        self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 0)
        self.assertEqual(torch.cuda.FloatTensor(1, device=1).get_device(), 1)
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 1)
            self.assertEqual(torch.cuda.FloatTensor(1, device=0).get_device(), 0)
            self.assertEqual(torch.cuda.FloatTensor(1, device=None).get_device(), 1)

    def test_events(self):
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

    @staticmethod
    def _stream_synchronize(self, spin_time_cycles):
        s = torch.cuda.current_stream()
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        e_tik.record(s)
        torch.cuda._sleep(spin_time_cycles)
        e_tok.record(s)
        s.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_synchronize(self, spin_time_cycles):
        s = torch.cuda.current_stream()
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        e_tik.record(s)
        torch.cuda._sleep(spin_time_cycles)
        s.record_event(e_tok)
        e_tok.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_wait(self, spin_time_cycles):
        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.Stream()
        e_tik = torch.cuda.Event(blocking=True, enable_timing=True)
        e_tok = torch.cuda.Event(blocking=True, enable_timing=True)

        e_tik.record(s0)
        torch.cuda._sleep(spin_time_cycles - 10)
        e_sync = torch.cuda.Event(blocking=True)
        e_sync.record()
        e_sync.wait(s1)
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10)
        s1.synchronize()
        s1.record_event(e_tok)

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())
        self.assertTrue(e_sync.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _test_stream_event_nogil(self, sync_func, p2c, c2p):
        with torch.cuda.device('cuda:1'):
            c2p.put(0)
            p2c.get()
            c2p.put(sync_func(self, TestCuda.FIFTY_MIL_CYCLES))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipIfRocm
    def test_stream_event_nogil(self):
        for sync_func in [TestCuda._stream_synchronize,
                          TestCuda._event_synchronize,
                          TestCuda._event_wait]:
            p2c = queue.Queue()
            c2p = queue.Queue()
            e_tik = torch.cuda.Event(enable_timing=True)
            e_tok = torch.cuda.Event(enable_timing=True)

            t = threading.Thread(
                target=TestCuda._test_stream_event_nogil,
                args=(self, sync_func, p2c, c2p))
            t.daemon = True
            t.start()

            c2p.get()
            with torch.cuda.device('cuda:0'):
                e_tik.record()
                p2c.put(0)
                parent_time = sync_func(self, TestCuda.FIFTY_MIL_CYCLES)
                child_time = c2p.get()
                e_tok.record()
                e_tok.synchronize()
                total_time = e_tik.elapsed_time(e_tok)

            # Without GIL, synchronizations in parent and child threads can
            # overlap. The total execution time should be a little bit longer
            # than spinning fifty million cycles and much shorter than twice of
            # that. However, testing absolute execution time is not reliable as
            # it may vary on different hardware in different environments.
            # Therefore, this test uses relative comparisons, checking if the
            # sum of parent and child threads execution time is greater than the
            # real execution time by least 40%.
            self.assertGreater(parent_time + child_time, total_time * 1.4)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_events_wait(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            e0 = torch.cuda.Event()
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()

        self.assertFalse(s0.query())
        self.assertTrue(s1.query())

        s1.wait_event(e0)
        s1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipIfRocm
    def test_events_multi_gpu_query(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = s0.record_event()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            e1 = s1.record_event()

        self.assertTrue(e0.query())
        self.assertFalse(e1.query())

        with torch.cuda.device(d0):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        with torch.cuda.device(d1):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        # deliberately using a different device
        with torch.cuda.device(d0):
            e1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(e1.query())

        with torch.cuda.device(d0):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

        with torch.cuda.device(d1):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipIfRocm
    def test_events_multi_gpu_elapsed_time(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(10)
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            s1.record_event(e1)

        e0.synchronize()
        e1.synchronize()
        with torch.cuda.device(d0):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch.cuda.device(d1):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e2 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            s0.record_event(e2)
            s0.synchronize()

        self.assertGreater(e0.elapsed_time(e2), 0)

        # deliberately calling from a different device
        with torch.cuda.device(d1):
            self.assertGreater(e0.elapsed_time(e2), 0)

    def test_record_stream(self):
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.cuda.FloatTensor(t.size())
        stream = torch.cuda.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            with torch.cuda.stream(stream):
                tmp = t.cuda(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.cuda.current_stream().wait_stream(stream)
            tmp.record_stream(torch.cuda.current_stream())
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with torch.cuda.stream(stream):
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(tmp2.data_ptr(), ptr[0], 'allocation re-used to soon')

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        # Check that the block will be re-used after the main stream finishes
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(stream):
            tmp3 = torch.cuda.FloatTensor(t.size())
            self.assertEqual(tmp3.data_ptr(), ptr[0], 'allocation not re-used')

    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        cycles_per_ms = get_cycles_per_ms()

        # check that allocations are re-used after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, 'allocation not reused')

        # check that the allocation is not re-used if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')
        self.assertEqual(list(gpu_tensor), [1])

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_caching_pinned_memory_multi_gpu(self):
        # checks that the events preventing pinned memory from being re-used
        # too early are recorded on the correct GPU
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        gpu_tensor0 = torch.cuda.FloatTensor([0], device=0)
        gpu_tensor1 = torch.cuda.FloatTensor([0], device=1)

        with torch.cuda.device(1):
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            gpu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')

        with torch.cuda.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(gpu_tensor1[0], 1)
        self.assertEqual(gpu_tensor0[0], 2)

    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cuda_malloc_retry. see issue #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device='cuda')

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device='cuda')
            with torch.cuda.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.cuda.empty_cache()

    def test_reduction_gpu_memory_accessing(self):
        x = torch.ones(512, 8, dtype=torch.float32, device='cuda')
        torch.sum(x, 0)

    def test_sum_cpu_gpu_mismatch(self):
        x = torch.randn(20, dtype=torch.float32, device='cuda:0')
        y = torch.randn(1, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError,
                                    'expected device cpu and dtype Float but got device cuda:0 and dtype Float'):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)
        # makeing sure half to float promotion is also properly working.
        x = x.half()
        with self.assertRaisesRegex(RuntimeError,
                                    'expected device cpu and dtype Float but got device cuda:0 and dtype Half'):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)

    @skipIfRocm
    def test_sum_noncontig(self):
        x = torch.randn(1, 75, 57, 20, device='cuda').permute(0, 3, 1, 2)
        y = x.cpu()
        self.assertEqual(x.sum().cpu(), y.sum())
        self.assertEqual(x.sum(dim=(-1, -2)).cpu(), y.sum(dim=(-1, -2)))
        self.assertEqual(x.sum(dim=(1, 3)).cpu(), y.sum(dim=(1, 3)))

    def test_sum_fp16(self):
        x = torch.zeros(10, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        x = torch.ones(65504, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(), 65504)
        self.assertEqual(x.sum(dtype=torch.float32), 65504)

        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    def test_mean_fp16(self):
        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.mean(dtype=torch.float32), 1)

    def test_prod_large(self):
        # tests global reduction (should_global_reduce = true) in case of non-zero identity element
        x = torch.ones(240000, device='cuda', dtype=torch.float32)
        self.assertEqual(x.prod(), 1)

    @staticmethod
    def _select_broadcastable_dims(dims_full=None):
        return _TestTorchMixin._select_broadcastable_dims(dims_full)

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_inverse(self):
        _TestTorchMixin._test_inverse(self, lambda t: t.cuda())

    @slowTest
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_inverse_many_batches(self):
        _TestTorchMixin._test_inverse_slow(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_pinverse(self):
        _TestTorchMixin._test_pinverse(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_matrix_rank(self):
        _TestTorchMixin._test_matrix_rank(self, lambda x: x.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_matrix_power(self):
        _TestTorchMixin._test_matrix_power(self, conv_fn=lambda t: t.cuda())

    def test_chain_matmul(self):
        _TestTorchMixin._test_chain_matmul(self, cast=lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_det_logdet_slogdet(self):
        _TestTorchMixin._test_det_logdet_slogdet(self, 'cuda')

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_solve(self):
        _TestTorchMixin._test_solve(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_solve_batched(self):
        _TestTorchMixin._test_solve_batched(self, lambda t: t.cuda())

    @slowTest
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_solve_batched_many_batches(self):
        _TestTorchMixin._test_solve_batched_many_batches(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_solve_batched_dims(self):
        _TestTorchMixin._test_solve_batched_dims(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_solve(self):
        _TestTorchMixin._test_cholesky_solve(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_solve_batched(self):
        _TestTorchMixin._test_cholesky_solve_batched(self, lambda t: t.cuda())

    @slowTest
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_solve_batched_many_batches(self):
        _TestTorchMixin._test_cholesky_solve_batched_many_batches(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_solve_batched_dims(self):
        _TestTorchMixin._test_cholesky_solve_batched_dims(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_inverse(self):
        _TestTorchMixin._test_cholesky_inverse(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky(self):
        _TestTorchMixin._test_cholesky(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_cholesky_batched(self):
        _TestTorchMixin._test_cholesky_batched(self, lambda t: t.cuda())

    def test_view(self):
        _TestTorchMixin._test_view(self, lambda t: t.cuda())

    def test_flip(self):
        _TestTorchMixin._test_flip(self, use_cuda=True)

    def test_rot90(self):
        _TestTorchMixin._test_rot90(self, use_cuda=True)

    def test_signal_window_functions(self):
        _TestTorchMixin._test_signal_window_functions(self, device=torch.device('cuda'))

    @skipIfRocm
    def test_fft_ifft_rfft_irfft(self):
        _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        @contextmanager
        def plan_cache_max_size(n, device=None):
            if device is None:
                plan_cache = torch.backends.cuda.cufft_plan_cache
            else:
                plan_cache = torch.backends.cuda.cufft_plan_cache[device]
            original = plan_cache.max_size
            plan_cache.max_size = n
            yield
            plan_cache.max_size = original

        with plan_cache_max_size(max(1, torch.backends.cuda.cufft_plan_cache.size - 10)):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        with plan_cache_max_size(0):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        torch.backends.cuda.cufft_plan_cache.clear()

        # check that stll works after clearing cache
        with plan_cache_max_size(10):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        with self.assertRaisesRegex(RuntimeError, r"must be non-negative"):
            torch.backends.cuda.cufft_plan_cache.max_size = -1

        with self.assertRaisesRegex(RuntimeError, r"read-only property"):
            torch.backends.cuda.cufft_plan_cache.size = -1

        with self.assertRaisesRegex(RuntimeError, r"but got device with index"):
            torch.backends.cuda.cufft_plan_cache[torch.cuda.device_count() + 10]

        if TEST_MULTIGPU:
            # Test that different GPU has different cache
            x0 = torch.randn(2, 3, 3, device='cuda:0')
            x1 = x0.cuda(1)
            self.assertEqual(x0.rfft(2), x1.rfft(2))
            # If a plan is used across different devices, the following line (or
            # the assert above) would trigger illegal memory access. Other ways
            # to trigger the error include
            #   (1) setting CUDA_LAUNCH_BLOCKING=1 (pytorch/pytorch#19224) and
            #   (2) printing a device 1 tensor.
            x0.copy_(x1)

            # Test that un-indexed `torch.backends.cuda.cufft_plan_cache` uses current device
            with plan_cache_max_size(10, device='cuda:0'):
                with plan_cache_max_size(11, device='cuda:1'):
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                    self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                    with torch.cuda.device(1):
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(0):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0

                self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                with torch.cuda.device(1):
                    with plan_cache_max_size(11):  # default is cuda:1
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(0):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1

    def test_stft(self):
        _TestTorchMixin._test_stft(self, device=torch.device('cuda'))

    def test_multinomial(self):
        _TestTorchMixin._test_multinomial(self, torch.cuda.FloatTensor)

        # Test two corner cases from older PyTorch (Issue #4858)
        freqs = torch.cuda.FloatTensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.03178183361887932, 0.027680952101945877, 0.033176131546497345,
            0.046052902936935425, 0.07742464542388916, 0.11543981730937958,
            0.14148041605949402, 0.15784293413162231, 0.13180233538150787,
            0.08271478116512299, 0.049702685326337814, 0.027557924389839172,
            0.018125897273421288, 0.011851548217236996, 0.010252203792333603,
            0.007422595750540495, 0.005372154992073774, 0.0045109698548913,
            0.0036087757907807827, 0.0035267581697553396, 0.0018864056328311563,
            0.0024605290964245796, 0.0022964938543736935, 0.0018453967059031129,
            0.0010662291897460818, 0.0009842115687206388, 0.00045109697384759784,
            0.0007791675161570311, 0.00020504408166743815, 0.00020504408166743815,
            0.00020504408166743815, 0.00012302644609007984, 0.0,
            0.00012302644609007984, 4.100881778867915e-05, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0])

        torch.cuda.manual_seed(11042)
        sample = torch.multinomial(freqs, 1000, True)
        self.assertNotEqual(freqs[sample].min(), 0)

        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        p[:, 1] = 1
        torch.cuda.manual_seed(5214)
        r = torch.multinomial(p, 1)
        self.assertNotEqual(r.min().item(), 0)

        # test corner case from Issue #13867
        torch.cuda.manual_seed(33)
        probs = torch.randn(1000000, device='cuda').clamp(min=0) * 3e-5
        samples = probs.multinomial(1000000, replacement=True)
        self.assertGreater(probs[samples].min().item(), 0)

    @skipCUDANonDefaultStreamIf(True)
    def test_multinomial_alias(self):
        _TestTorchMixin._test_multinomial_alias(self, lambda t: t.cuda())

    @staticmethod
    def mute():
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())

    def _spawn_method(self, method, arg):
        ctx = mp.get_context("spawn")
        with ctx.Pool(1, initializer=self.mute) as pool:
            errors = pool.map(method, [arg])
            for e in errors:
                if 'device-side assert triggered' not in str(e):
                    self.fail(e)

    @staticmethod
    def _test_multinomial_invalid_probs_cuda(probs):
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.multinomial(probs.to('cuda'), 2)
                torch.cuda.synchronize()
            return False  # Should not be reached
        except RuntimeError as e:
            return e

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(IS_WINDOWS, 'FIXME: CUDA OOM error on Windows')
    @unittest.skipIf(not PY3,
                     "spawn start method is not supported in Python 2, \
                     but we need it for creating another process with CUDA")
    def test_multinomial_invalid_probs_cuda(self):
        test_method = TestCuda._test_multinomial_invalid_probs_cuda
        self._spawn_method(test_method, torch.Tensor([1, -1, 1]))
        self._spawn_method(test_method, torch.Tensor([1, inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, -inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, 1, nan]))
        self._spawn_method(test_method, torch.Tensor([0, 1, 0]))

    def test_broadcast(self):
        _TestTorchMixin._test_broadcast(self, lambda t: t.cuda())

    def test_contiguous(self):
        _TestTorchMixin._test_contiguous(self, lambda t: t.cuda())

    def test_broadcast_fused_matmul(self):
        _TestTorchMixin._test_broadcast_fused_matmul(self, lambda t: t.cuda())

    def test_broadcast_batched_matmul(self):
        _TestTorchMixin._test_broadcast_batched_matmul(self, lambda t: t.cuda())

    def test_index(self):
        _TestTorchMixin._test_index(self, lambda t: t.cuda())

    @skipCUDANonDefaultStreamIf(True)
    def test_advancedindex(self):
        _TestTorchMixin._test_advancedindex(self, lambda t: t.cuda())

    def test_advancedindex_mixed_cpu_cuda(self):
        def test(x, ia, ib):
            # test getitem
            self.assertEqual(x[:, ia, None, ib, 0].cpu(),
                             x.cpu()[:, ia.cpu(), None, ib.cpu(), 0])
            self.assertEqual(x[ia], x.cpu()[ia.cpu()])
            # test setitem
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            first_shape = x[:, ia, None, ib, 0].shape
            second_shape = x[ia].shape
            x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
            x_clone2[ia] = torch.randn(second_shape).to(x_clone2)

        cpu = torch.device('cpu')
        for device in ['cuda:0', 'cuda:1'] if torch.cuda.device_count() > 1 else ['cuda']:
            # Index cpu tensor with cuda tensor
            x = torch.randn(3, 4, 4, 4, 3)
            ia = torch.tensor([0, 2, 1]).to(device)
            ib = torch.tensor([0, 2, 1]).to(device)
            test(x, ia, ib)

            # Index cuda tensor with cpu tensor
            x = x.to(device)
            ia = ia.to(cpu)
            ib = ib.to(cpu)
            test(x, ia, ib)

            # Index cpu tensor with mixed cpu, cuda tensors
            x = x.to(cpu)
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x, ia, ib)

            # Index cuda tensor with mixed cpu, cuda tensors
            x = x.to(device)
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x, ia, ib)

            if torch.cuda.device_count() > 1:
                other_device = 'cuda:0' if device != 'cuda:0' else 'cuda:1'
                # Index cuda tensor with mixed cpu, cuda tensors on different devices
                x = x.to(device)
                ia = ia.to(cpu)
                ib = ib.to(other_device)
                test(x, ia, ib)

    def test_advancedindex_big(self):
        _TestTorchMixin._test_advancedindex_big(self, lambda t: t.cuda())

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_huge_index(self):
        src = torch.empty(15000000, 45, device='cuda', dtype=torch.long).random_(0, 2**22)
        idx = torch.randperm(src.shape[0], device='cuda')
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_kthvalue(self):
        _TestTorchMixin._test_kthvalue(self, device='cuda')

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_lu(self):
        _TestTorchMixin._test_lu(self, lambda t: t.cuda(), pivot=False)
        _TestTorchMixin._test_lu(self, lambda t: t.cuda(), pivot=True)

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_lu_solve(self):
        _TestTorchMixin._test_lu_solve(self, lambda t: t.cuda(), pivot=False)
        _TestTorchMixin._test_lu_solve(self, lambda t: t.cuda(), pivot=True)

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_lu_unpack(self):
        _TestTorchMixin._test_lu_unpack(self, lambda t: t.cuda(), pivot=False)
        _TestTorchMixin._test_lu_unpack(self, lambda t: t.cuda(), pivot=True)

    def test_dim_reduction(self):
        _TestTorchMixin._test_dim_reduction(self, lambda t: t.cuda())

    def test_tensor_gather(self):
        _TestTorchMixin._test_gather(self, lambda t: t.cuda(), False)

    def test_tensor_scatter(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', test_bounds=False)

    def test_tensor_scatterAdd(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_add_', test_bounds=False)

    def test_tensor_scatterFill(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', True, test_bounds=False)

    def test_min_max_inits(self):
        # Testing if THC_reduceAll received the correct index initialization.
        # This affects the result of THC_reduceAll operations at extreme values
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

    def test_max_with_inf(self):
        _TestTorchMixin._test_max_with_inf(self, (torch.half, torch.float, torch.double), 'cuda')

    def test_min_with_inf(self):
        _TestTorchMixin._test_min_with_inf(self, (torch.half, torch.float, torch.double), 'cuda')

    def test_rpow(self):
        _TestTorchMixin._test_rpow(self, lambda x: x.cuda())

    def test_int_pow(self):
        _TestTorchMixin._test_int_pow(self, lambda x: x.cuda())

    def test_remainder_overflow(self):
        _TestTorchMixin._test_remainder_overflow(self, dtype=torch.int64, device='cuda')

    def test_var(self):
        cpu_tensor = torch.randn(2, 3, 3)
        gpu_tensor = cpu_tensor.cuda()
        self.assertEqual(gpu_tensor.var(), cpu_tensor.var())
        self.assertEqual(gpu_tensor.var(1), cpu_tensor.var(1))
        self.assertEqual(gpu_tensor.var(2), cpu_tensor.var(2))
        self.assertEqual(gpu_tensor.std(), cpu_tensor.std())
        self.assertEqual(gpu_tensor.std(1), cpu_tensor.std(1))
        self.assertEqual(gpu_tensor.var(2), cpu_tensor.var(2))

        cpu_tensor = torch.randn(100)
        gpu_tensor = cpu_tensor.cuda()
        self.assertEqual(gpu_tensor.var(), cpu_tensor.var())

    def test_var_unbiased(self):
        tensor = torch.randn(100).cuda()
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False))

        tensor = torch.FloatTensor([1.0, 2.0]).cuda()
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.randn(100).cuda()
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False))

    def test_var_large_input(self):
        # Large, not-nice input
        tensor_cpu = torch.randn(2 * 32 * 1024 + 1, 2, 67)
        tensor_cuda = tensor_cpu.cuda()

        self.assertEqual(tensor_cpu.var(2), tensor_cuda.var(2).cpu())

    def test_var_stability(self):
        tensor = torch.FloatTensor([2281.5, 2281.25]).cuda()

        # Stability for inner dim
        self.assertEqual(tensor.var(0), 0.03125)

        # General stability
        self.assertEqual(tensor.var(), 0.03125)

        # Stability for outer dimensions
        tensor = tensor.unsqueeze(1)
        self.assertEqual(tensor.var(0), 0.03125)

    @skipIfRocm
    def test_digamma(self):
        def test(use_double=False):
            cpu_tensor = torch.randn(10, 10, 10)
            gpu_tensor = cpu_tensor.cuda()
            zeros = torch.zeros(10, 10, 10)
            if (use_double):
                cpu_tensor = cpu_tensor.double()
                gpu_tensor = gpu_tensor.double()
                zeros = zeros.double()
            cpu_out = cpu_tensor.digamma()
            gpu_out = gpu_tensor.digamma()
            norm_errors = (gpu_out - cpu_out.cuda()) / gpu_out
            self.assertEqual(norm_errors, zeros)

        test(True)
        test(False)

        # Test float32 behavior near and at poles.
        cpu_tensor = torch.tensor([-0.999999994, -1.999999994, -2.0000000111,
                                  -100.99999994, -1931.99999994, 0.000000111,
                                  -0.000000111, 0, -1, -2, -931])
        expected_errors = torch.tensor([0, 0, 0, 0, 0, 0, 0, nan, nan, nan, nan])
        gpu_tensor = cpu_tensor.cuda()
        cpu_out = cpu_tensor.digamma()
        gpu_out = gpu_tensor.digamma()
        norm_errors = (gpu_out - cpu_out.cuda()) / gpu_out
        self.assertEqual(norm_errors, expected_errors)

    @skipIfRocm
    def test_polygamma(self):
        def test(use_double=False):
            cpu_tensor = torch.randn(10, 10, 10)
            gpu_tensor = cpu_tensor.cuda()
            zeros = torch.zeros(10, 10, 10)
            if (use_double):
                cpu_tensor = cpu_tensor.double()
                gpu_tensor = gpu_tensor.double()
                zeros = zeros.double()
            for n in [0, 1]:
                cpu_out = cpu_tensor.polygamma(n)
                gpu_out = gpu_tensor.polygamma(n)
                norm_errors = (gpu_out - cpu_out.cuda()) / gpu_out
                self.assertEqual(norm_errors, zeros)

        test(True)
        test(False)

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_symeig(self):
        _TestTorchMixin._test_symeig(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_svd(self):
        _TestTorchMixin._test_svd(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_svd_no_singularvectors(self):
        _TestTorchMixin._test_svd_no_singularvectors(self, lambda t: t.cuda())

    def test_arange(self):
        for t in ['IntTensor', 'LongTensor', 'FloatTensor', 'DoubleTensor']:
            a = torch.cuda.__dict__[t]()
            torch.arange(0, 10, out=a)
            b = torch.__dict__[t]()
            torch.arange(0, 10, out=b)
            self.assertEqual(a, b.cuda())

    def test_linspace(self):
        a = torch.linspace(0, 10, 10, device='cuda')
        b = torch.linspace(0, 10, 10)
        self.assertEqual(a, b.cuda())

    def test_logspace(self):
        a = torch.logspace(1, 10, 10, device='cuda')
        b = torch.logspace(1, 10, 10)
        self.assertEqual(a, b.cuda())

        # Check non-default base=2
        a = torch.logspace(1, 10, 10, 2, device='cuda')
        b = torch.logspace(1, 10, 10, 2)
        self.assertEqual(a, b.cuda())

    def test_lerp(self):
        _TestTorchMixin._test_lerp(self, lambda t: t.cuda())

    def test_diagonal(self):
        _TestTorchMixin._test_diagonal(self, dtype=torch.float32, device='cuda')

    def test_diagflat(self):
        _TestTorchMixin._test_diagflat(self, dtype=torch.float32, device='cuda')

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    @skipCUDANonDefaultStreamIf(True)
    def test_norm(self):
        _TestTorchMixin._test_norm(self, device='cuda')

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    @skipCUDANonDefaultStreamIf(True)
    def test_nuclear_norm_axes_small_brute_force(self):
        _TestTorchMixin._test_nuclear_norm_axes(self, device='cuda')

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    @skipCUDANonDefaultStreamIf(True)
    def test_nuclear_norm_exceptions(self):
        _TestTorchMixin._test_nuclear_norm_exceptions(self, device='cuda')

    def test_dist(self):
        _TestTorchMixin._test_dist(self, device='cuda')

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_geqrf(self):
        _TestTorchMixin._test_geqrf(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    @skipCUDANonDefaultStreamIf(True)
    def test_triangular_solve(self):
        _TestTorchMixin._test_triangular_solve(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_triangular_solve_batched(self):
        _TestTorchMixin._test_triangular_solve_batched(self, lambda t: t.cuda())

    @slowTest
    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_triangular_solve_batched_many_batches(self):
        _TestTorchMixin._test_triangular_solve_batched_many_batches(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_triangular_solve_batched_dims(self):
        _TestTorchMixin._test_triangular_solve_batched_dims(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MAGMA, "no MAGMA library detected")
    def test_qr(self):
        _TestTorchMixin._test_qr(self, lambda t: t.cuda())

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_get_set_rng_state_all(self):
        states = torch.cuda.get_rng_state_all()
        before0 = torch.cuda.FloatTensor(100, device=0).normal_()
        before1 = torch.cuda.FloatTensor(100, device=1).normal_()
        torch.cuda.set_rng_state_all(states)
        after0 = torch.cuda.FloatTensor(100, device=0).normal_()
        after1 = torch.cuda.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, 0)
        self.assertEqual(before1, after1, 0)

    @skipIfRocm
    def test_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()

    def test_randperm_cuda(self):
        cuda = torch.device('cuda:0')

        # Test core functionality. For small n, randperm is offloaded to CPU instead. For large n, randperm is executed
        # on GPU.
        for n in (100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on cuda.
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                with torch.random.fork_rng(devices=[0]):
                    res1 = torch.randperm(n, dtype=dtype, device=cuda)
                res2 = torch.empty(0, dtype=dtype, device=cuda)
                torch.randperm(n, out=res2, dtype=dtype, device=cuda)
                self.assertEqual(res1, res2, 0)

        # Default type is long
        for n in (100, 50000):
            self.assertIsInstance(torch.randperm(n, device=cuda), torch.cuda.LongTensor)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0, device=cuda)
        res2 = torch.cuda.LongTensor(5)
        torch.randperm(0, out=res2, device=cuda)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for res, small_n, large_n in ((torch.cuda.HalfTensor(), 2**11 + 1, 2**11 + 2),
                                      (torch.cuda.FloatTensor(), 2**24 + 1, 2**24 + 2),
                                      (torch.cuda.DoubleTensor(), 2**25,  # 2**53 + 1 is too large to run
                                       2**53 + 2)):
            torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(large_n, out=res))

    def test_random_neg_values(self):
        _TestTorchMixin._test_random_neg_values(self, use_cuda=True)

    def test_bincount_cuda(self):
        _TestTorchMixin._test_bincount(self, device='cuda')
        # ensure CUDA code coverage
        input_size = (5000,)
        w = torch.randn(input_size, device='cuda')
        w_cpu = w.cpu()
        # test shared memory impl
        t = torch.randint(50, input_size, dtype=torch.int8, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test multi block memory impl
        # see `THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM` in SummaryOps.cu
        t = torch.randint(500, input_size, dtype=torch.int64, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test global memory impl
        # see `THRESH_NUMBER_BINS_FOR_GLOBAL_MEM` in SummaryOps.cu
        t = torch.randint(2000, input_size, dtype=torch.int64, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

    def test_tiny_half_norm_(self):
        a = torch.arange(25).cuda().float()
        a /= 100000000
        b = a.half()
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        a = torch.ones(65536).cuda().half()
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    # Note: This test fails on ROCm CI gfx900 but passes on gfx906
    @skipIfRocm
    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_cuda_memory_leak_detection(self):
        l = []

        @self.wrap_with_cuda_memory_check
        def no_leak():
            pass

        @self.wrap_with_cuda_memory_check
        def leak_gpu0():
            l.append(torch.tensor(10, device=torch.device("cuda:0")))

        no_leak()

        with self.assertRaisesRegex(AssertionError, r"leaked \d+ bytes CUDA memory on device 0"):
            leak_gpu0()

        if TEST_MULTIGPU:
            @self.wrap_with_cuda_memory_check
            def leak_gpu1():
                l.append(torch.tensor(10, device=torch.device("cuda:1")))

            with self.assertRaisesRegex(AssertionError, r"leaked \d+ bytes CUDA memory on device 1"):
                leak_gpu1()

    def test_cuda_memory_leak_detection_propagates_errors(self):
        with self.assertRaisesRegex(RuntimeError, r"The size of tensor a \(3\) must match"):
            with self.assertLeaksNoCudaTensors():
                x = torch.randn(3, 1, device='cuda')
                y = torch.randn(2, 1, device='cuda')
                z = x + y

    def test_trilu_indices(self):
        for test_args in tri_tests_args:
            _compare_trilu_indices(self, *test_args, device='cuda')

        # test default options
        x = torch.ones(
            3, 3, dtype=torch.long, device='cuda', layout=torch.strided)
        self.assertEqual(
            x.tril(0).nonzero().transpose(0, 1),
            torch.tril_indices(3, 3, device='cuda'))
        self.assertEqual(
            x.triu(0).nonzero().transpose(0, 1),
            torch.triu_indices(3, 3, device='cuda'))

    def test_large_trilu_indices(self):
        for test_args in tri_large_tests_args:
            _compare_large_trilu_indices(self, *test_args, device='cuda')

    def test_triu_tril(self):
        _TestTorchMixin._test_triu_tril(self, lambda t: t.cuda())

    def test_cuda_round(self):
        # test half-to-even
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        self.assertEqual(
            torch.HalfTensor(a).cuda().round().cpu(),
            torch.HalfTensor(res).cpu())
        self.assertEqual(
            torch.FloatTensor(a).cuda().round().cpu(),
            torch.FloatTensor(res).cpu())
        self.assertEqual(
            torch.DoubleTensor(a).cuda().round().cpu(),
            torch.DoubleTensor(res).cpu())


def load_ignore_file():
    from os.path import join, dirname
    global ignores
    path = join(dirname(__file__), 'data', 'test_cuda_ignores.txt')
    with open(path, 'r') as f:
        ignores = {l for l in f.read().splitlines() if not l.startswith('#')}


def generate_tests():
    for decl in tests:
        for t in types:
            tensor = t()

            # Default values
            desc = ''
            type_subset = types
            no_inplace = False
            decorator = None
            if len(decl) == 3:
                name, constr, arg_constr = decl
            elif len(decl) == 4:
                name, constr, arg_constr, desc = decl
            elif len(decl) == 5:
                name, constr, arg_constr, desc, type_subset = decl
            elif len(decl) == 6:
                name, constr, arg_constr, desc, type_subset, no_inplace = decl
            elif len(decl) == 7:
                name, constr, arg_constr, desc, type_subset, no_inplace, decorator = decl

            if t not in type_subset:
                continue
            if TEST_WITH_ROCM and decorator is not None:
                if (isinstance(decorator, str)):
                    tensor_type_name = str(t.__name__)
                    decorator_list = decorator.split(":")
                    skip_type_list = decorator_list[1].split(",")
                    if (("ByteTensor" in skip_type_list) and tensor_type_name == "ByteTensor") \
                            or (("CharTensor" in skip_type_list) and tensor_type_name == "CharTensor") \
                            or (("DoubleTensor" in skip_type_list) and tensor_type_name == "DoubleTensor") \
                            or (("FloatTensor" in skip_type_list) and tensor_type_name == "FloatTensor") \
                            or (("HalfTensor" in skip_type_list) and tensor_type_name == "HalfTensor") \
                            or (("IntTensor" in skip_type_list) and tensor_type_name == "IntTensor") \
                            or (("LongTensor" in skip_type_list) and tensor_type_name == "LongTensor") \
                            or (("ShortTensor" in skip_type_list) and tensor_type_name == "ShortTensor"):
                        decorator = skipIfRocm
                    else:
                        decorator = None
            elif ((not TEST_WITH_ROCM) and (decorator is not None)):
                if (isinstance(decorator, str)):
                    decorator = None

            precision = custom_precision.get(name, TestCuda.precision)
            if is_half(t):
                precision = custom_half_precision.get(name, precision)

            for inplace in (True, False):
                if inplace and no_inplace:
                    continue
                if inplace:
                    name_inner = name + '_'
                else:
                    name_inner = name

                if t != torch.HalfTensor and not hasattr(tensor, name_inner):
                    # torch.HalfTensor doesn't support most operations,
                    # but we use torch.FloatTensor as cpu baseline
                    continue
                full_name = '{}.{}'.format(tensor.type(), name_inner)
                if full_name in ignores:
                    continue

                test_name = 'test_' + t.__name__ + '_' + name_inner
                if desc:
                    test_name += '_' + desc

                assert not hasattr(TestCuda, test_name), "Duplicated test name: " + test_name

                test_fn = compare_cpu_gpu(constr, arg_constr, name_inner, t, precision)

                if decorator is not None:
                    test_fn = decorator(test_fn)

                setattr(TestCuda, test_name, test_fn)


if __name__ == '__main__':
    if TEST_CUDA:
        load_ignore_file()
        generate_tests()

    run_tests()
