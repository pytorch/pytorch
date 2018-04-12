import math
import tempfile
import re
import unittest
from itertools import repeat

import torch
import torch.cuda
import torch.cuda.comm as comm

from test_torch import TestTorch
from common import TestCase, get_gpu_type, to_gpu, freeze_rng_state, run_tests

HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811
    HAS_CUDA = False

HAS_MAGMA = HAS_CUDA
if HAS_CUDA:
    torch.ones(1).cuda()  # has_magma shows up after cuda is initialized
    HAS_MAGMA = torch.cuda.has_magma

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
    name = type(t).__name__
    if 'Double' in name or 'Float' in name or 'Half' in name:
        return floating
    else:
        return integer

S = 10
M = 50


def make_tensor(t, *sizes):
    return t(*sizes).copy_(torch.randn(*sizes))


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
tests = [
    ('add', small_3d, lambda t: [number(3.14, 3, t)]),
    ('add', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('add', small_3d, lambda t: [number(0.2, 2, t), small_3d_positive(t)], 'scalar_tensor'),
    ('sub', small_3d, lambda t: [number(3.14, 3, t)],),
    ('sub', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('mul', small_3d, lambda t: [number(3.14, 3, t)],),
    ('mul', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('div', small_3d, lambda t: [number(3.14, 3, t)],),
    ('div', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('pow', small_3d, lambda t: [number(3.14, 3, t)], None, float_types),
    ('pow', small_3d, lambda t: [number(1., 1, t)], 'pow1', types),
    ('pow', small_3d, lambda t: [number(2., 2, t)], 'pow2', types),
    ('pow', small_3d, lambda t: [number(3., 3, t)], 'pow3', types),
    ('pow', small_3d, lambda t: [number(-1., -1, t)], 'pow-1', float_types),
    # HalfTensor gives bad result at pow-2 with data sampled from torch.randn
    ('pow', small_3d, lambda t: [number(-2., -2, t)], 'pow-2', float_types_no_half),
    ('pow', small_3d, lambda t: [tensor_abs_(small_3d(t))], 'tensor', float_types),
    ('addbmm', small_2d, lambda t: [small_3d(t), small_3d(t)], None, float_types),
    ('addbmm', small_2d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('addbmm', small_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars'),
    ('baddbmm', small_3d, lambda t: [small_3d(t), small_3d(t)],),
    ('baddbmm', small_3d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('baddbmm', small_3d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars'),
    ('addcdiv', small_2d_lapack, lambda t: [tensor_mul(small_2d_lapack(t), 2), small_2d_lapack(t)],),
    ('addcdiv', small_2d_lapack, lambda t: [number(2.8, 1, t),
                                            tensor_mul(small_2d_lapack(t), 2), small_2d_lapack(t)], 'scalar'),
    ('addcmul', small_3d, lambda t: [small_3d(t), small_3d(t)],),
    ('addcmul', small_3d, lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar'),
    ('addmm', medium_2d, lambda t: [medium_2d(t), medium_2d(t)],),
    ('addmm', medium_2d, lambda t: [number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'scalar'),
    ('addmm', medium_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'two_scalars'),
    ('addmv', medium_1d, lambda t: [medium_2d(t), medium_1d(t)],),
    ('addmv', medium_1d, lambda t: [number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'scalar'),
    ('addmv', medium_1d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'two_scalars'),
    ('addr', medium_2d, lambda t: [medium_1d(t), medium_1d(t)],),
    ('addr', medium_2d, lambda t: [number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'scalar'),
    ('addr', medium_2d, lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'two_scalars'),
    ('atan2', medium_2d, lambda t: [medium_2d(t)], None, float_types + [torch.HalfTensor]),
    ('fmod', small_3d, lambda t: [3], 'value'),
    ('fmod', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('chunk', medium_2d, lambda t: [4],),
    ('chunk', medium_2d, lambda t: [4, 1], 'dim'),
    ('chunk', medium_2d, lambda t: [4, -2], 'neg_dim'),
    ('clamp', medium_2d_scaled, lambda t: [-1, 5], None, signed_types),
    ('clamp', medium_2d_scaled, lambda t: [1, 5], None, unsigned_types),
    ('clone', medium_2d, lambda t: [],),
    ('contiguous', medium_2d, lambda t: [],),
    ('cross', new_t(M, 3, M), lambda t: [new_t(M, 3, M)(t)],),
    ('cumprod', small_3d, lambda t: [1],),
    ('cumprod', small_3d, lambda t: [-1], 'neg_dim'),
    ('cumsum', small_3d, lambda t: [1],),
    ('cumsum', small_3d, lambda t: [-1], 'neg_dim'),
    ('dim', small_3d, lambda t: [],),
    ('dist', small_2d, lambda t: [small_2d(t)],),
    ('dist', small_2d, lambda t: [small_2d(t), 3], '3_norm'),
    ('dist', small_2d, lambda t: [small_2d(t), 2.5], '2_5_norm'),
    ('dot', medium_1d, lambda t: [medium_1d(t)],),
    ('element_size', medium_1d, lambda t: [],),
    ('eq', small_3d_ones, lambda t: [small_3d(t)],),
    ('eq', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('ne', small_3d_ones, lambda t: [small_3d(t)],),
    ('ne', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('equal', small_3d_ones, lambda t: [small_3d_ones(t)], 'equal'),
    ('equal', small_3d_ones, lambda t: [small_3d(t)],),
    ('expand', new_t(M, 1, M), lambda t: [M, 4, M],),
    ('expand_as', new_t(M, 1, M), lambda t: [new_t(M, 4, M)(t)],),
    ('fill', medium_2d, lambda t: [number(3.14, 3, t)],),
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
    ('lerp', small_3d, lambda t: [small_3d(t), 0.3],),
    ('max', small_3d_unique, lambda t: [],),
    ('max', small_3d_unique, lambda t: [1], 'dim'),
    ('max', small_3d_unique, lambda t: [-1], 'neg_dim'),
    ('max', medium_2d, lambda t: [medium_2d(t)], 'elementwise'),
    ('min', small_3d_unique, lambda t: [],),
    ('min', small_3d_unique, lambda t: [1], 'dim'),
    ('min', small_3d_unique, lambda t: [-1], 'neg_dim'),
    ('min', medium_2d, lambda t: [medium_2d(t)], 'elementwise'),
    ('mean', small_3d, lambda t: [],),
    ('mean', small_3d, lambda t: [-1], 'neg_dim'),
    ('mean', small_3d, lambda t: [1], 'dim'),
    ('mode', small_3d, lambda t: [],),
    ('mode', small_3d, lambda t: [1], 'dim'),
    ('mode', small_3d, lambda t: [-1], 'neg_dim'),
    ('remainder', small_3d, lambda t: [3], 'value'),
    ('remainder', small_3d, lambda t: [-3], 'negative_value', signed_types),
    ('remainder', small_3d, lambda t: [small_3d_positive(t)], 'tensor'),
    ('remainder', small_3d, lambda t: [constant_tensor_sub(0, small_3d_positive(t))], 'negative_tensor', signed_types),
    ('std', small_3d, lambda t: [],),
    ('std', small_3d, lambda t: [1], 'dim'),
    ('std', small_3d, lambda t: [-1], 'neg_dim'),
    ('var', small_3d, lambda t: [],),
    ('var', small_3d, lambda t: [1], 'dim'),
    ('var', small_3d, lambda t: [-1], 'neg_dim'),
    ('ndimension', small_3d, lambda t: [],),
    ('nelement', small_3d, lambda t: [],),
    ('numel', small_3d, lambda t: [],),
    ('narrow', small_3d, lambda t: [1, 3, 2],),
    ('narrow', small_3d, lambda t: [-1, 3, 2], 'neg_dim'),
    ('nonzero', small_3d, lambda t: [],),
    ('norm', small_3d, lambda t: [],),
    ('norm', small_3d, lambda t: [3], '3_norm'),
    ('norm', small_3d, lambda t: [3, 0], '3_norm_dim'),
    ('norm', small_3d, lambda t: [3, -2], '3_norm_neg_dim'),
    ('ones', small_3d, lambda t: [1, 2, 3, 4, 5],),
    ('permute', new_t(1, 2, 3, 4), lambda t: [2, 1, 3, 0],),
    ('put_', new_t(2, 5, 3), lambda t: [long_type(t)([[0], [-2]]), t([[3], [4]])],),
    ('put_', new_t(2, 3), lambda t: [long_type(t)([]), t([])], 'empty'),
    ('put_', new_t(2, 2), lambda t: [long_type(t)([[1], [-3]]), t([[1], [2]]), True], 'accumulate'),
    ('prod', small_2d_oneish, lambda t: [],),
    ('prod', small_3d, lambda t: [1], 'dim'),
    ('prod', small_3d, lambda t: [-1], 'neg_dim'),
    ('sum', small_2d, lambda t: [],),
    ('sum', small_3d, lambda t: [1], 'dim'),
    ('sum', small_3d, lambda t: [-1], 'neg_dim'),
    ('renorm', small_3d, lambda t: [2, 1, 1], '2_norm'),
    ('renorm', small_3d, lambda t: [2, -1, 1], '2_norm_neg_dim'),
    ('renorm', small_3d, lambda t: [1.5, 1, 1], '1_5_norm'),
    ('repeat', small_2d, lambda t: [2, 2, 2],),
    ('size', new_t(1, 2, 3, 4), lambda t: [],),
    ('size', new_t(1, 2, 3, 4), lambda t: [1], 'dim'),
    ('size', new_t(1, 2, 3, 4), lambda t: [-2], 'neg_dim'),
    ('sort', small_3d_unique, lambda t: [],),
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
    ('take', new_t(3, 4), lambda t: [long_type(t)([[0], [-2]])],),
    ('transpose', new_t(1, 2, 3, 4), lambda t: [1, 2],),
    ('transpose', new_t(1, 2, 3, 4), lambda t: [-1, -2], 'neg_dim'),
    ('to_list', small_3d, lambda t: [],),
    ('topk', small_3d_unique, lambda t: [2, 1, False, True], 'dim_sort'),
    ('topk', small_3d_unique, lambda t: [2, -1, False, True], 'neg_dim_sort'),
    ('topk', small_3d_unique, lambda t: [2, 1, True, True], 'dim_desc_sort'),
    ('trace', medium_2d, lambda t: [],),
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
    ('view_as', small_3d, lambda t: [t(100, 10)],),
    ('zero', small_3d, lambda t: [],),
    ('zeros', small_3d, lambda t: [1, 2, 3, 4],),
    ('eye', small_2d, lambda t: [3, 4],),
    ('rsqrt', lambda t: constant_tensor_add(1, small_3d(t)), lambda t: [], None, float_types),
    ('sinh', lambda t: tensor_clamp(small_3d(t), -1, 1), lambda t: [], None, float_types),
    ('tan', lambda t: tensor_clamp(small_3d(t), -1, 1), lambda t: [], None, float_types),
    # lapack tests
    ('qr', small_2d_lapack, lambda t: [], 'square', float_types),
    ('qr', small_2d_lapack_skinny, lambda t: [], 'skinny', float_types),
    ('qr', small_2d_lapack_fat, lambda t: [], 'fat', float_types),
    ('qr', large_2d_lapack, lambda t: [], 'big', float_types),
    ('inverse', new_t(20, 20), lambda t: [], None, float_types),
    ('geqrf', new_t(20, 20), lambda t: [], None, float_types),
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
    'erfinv': 1e-3,
    'exp': 1e-2,
    'expm1': 1e-2,
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
        if t.__name__ == 'HalfTensor':
            cpu_tensor = cpu_tensor.float()
            cpu_args = [arg.float() if torch.is_tensor(arg) and is_half(arg) else arg for arg in cpu_args]
        cpu_result = getattr(cpu_tensor, fn)(*cpu_args)
        try:
            gpu_result = getattr(gpu_tensor, fn)(*gpu_args)
        except RuntimeError as e:
            reason = e.args[0]
            if 'only supports floating-point types' in reason or 'unimplemented data type' in reason:
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

        def assert_change(comp=1, empty_cache=False):
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

        assert_change(0)
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
            assert_change(1)
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
        assert_change(0)
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
        assert_change(-1)
        assert_change(0)
        self.assertEqual(torch.cuda.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1)
        self.assertEqual(torch.cuda.memory_allocated(device), m0)

        # test empty_cache
        assert_change(0, empty_cache=True)

    def test_memory_stats(self):
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            pass

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=1, N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=1, N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = float('inf')
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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
            self.assertEqual(y.cuda(-1).get_device(), 1)
        z = z.cuda()
        self.assertEqual(z.get_device(), 0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_new(self):
        x = torch.randn(3, 3).cuda()
        self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
        self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

        with torch.cuda.device(1):
            self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
            self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_type_conversions_same_gpu(self):
        x = torch.randn(5, 5).cuda(1)
        self.assertEqual(x.int().get_device(), 1)

    def test_neg(self):
        TestTorch._test_neg(self, lambda t: t.cuda())

    def _test_broadcast(self, input):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("only one GPU detected")
        result = comm.broadcast(input, (0, 1))
        for i, t in enumerate(result):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)

    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).cuda())

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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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
        if torch.cuda.device_count() < 2:
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
        if torch.cuda.device_count() < 2:
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
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_cat_autogpu(self):
        x = torch.randn(4, 4).cuda(1)
        y = torch.randn(4, 4).cuda(1)
        z = torch.cat([x, y], 0)
        self.assertEqual(z.get_device(), x.get_device())

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

    def test_cat_empty(self):
        TestTorch._test_cat_empty(self, use_cuda=True)

    def test_bernoulli(self):
        x = torch.tensor([0, 1], dtype=torch.float32, device='cuda')
        self.assertEqual(x.bernoulli().tolist(), [0, 1])

    def test_cat_bad_input_sizes(self):
        x = torch.randn(2, 1).cuda()
        y = torch.randn(2, 1, 1).cuda()
        z = torch.randn(2, 1, 1).cuda()
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z]))

        x = torch.randn(2, 1, 2).cuda()
        y = torch.randn(2, 1, 1).cuda()
        z = torch.randn(2, 2, 1).cuda()
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z], dim=1))

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

    @unittest.skipIf(torch.cuda.device_count() < 2, "detected only one GPU")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "detected only one GPU")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "detected only one GPU")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "detected only one GPU")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "detected only one GPU")
    def test_streams_multi_gpu(self):
        default_stream = torch.cuda.current_stream()
        self.assertEqual(default_stream.device, 0)
        stream = torch.cuda.Stream(device=1)
        self.assertEqual(stream.device, 1)
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.current_stream().device, 1)
            self.assertNotEqual(torch.cuda.current_stream(), default_stream)

    @unittest.skipIf(torch.cuda.device_count() < 2, "multi-GPU not supported")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
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

    @staticmethod
    def _select_broadcastable_dims(dims_full=None):
        return TestTorch._select_broadcastable_dims(dims_full)

    @unittest.skipIf(not HAS_MAGMA, "no MAGMA library detected")
    def test_det_logdet_slogdet(self):
        TestTorch._test_det_logdet_slogdet(self, lambda t: t.cuda())

    def test_view(self):
        TestTorch._test_view(self, lambda t: t.cuda())

    def test_fft_ifft_rfft_irfft(self):
        def cuda_randn_double(*sizes):
            return torch.cuda.DoubleTensor(*sizes).normal_()
        TestTorch._test_fft_ifft_rfft_irfft(self, build_fn=cuda_randn_double)

    def test_stft(self):
        def cuda_randn_double(*sizes):
            return torch.cuda.DoubleTensor(*sizes).normal_()
        TestTorch._test_stft(self, build_fn=cuda_randn_double)

    def test_multinomial(self):
        TestTorch._test_multinomial(self, torch.cuda.FloatTensor)

        # Test a corner case from older PyTorch (Issue #4858)
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

    def test_broadcast(self):
        TestTorch._test_broadcast(self, lambda t: t.cuda())

    def test_contiguous(self):
        TestTorch._test_contiguous(self, lambda t: t.cuda())

    def test_broadcast_fused_matmul(self):
        TestTorch._test_broadcast_fused_matmul(self, lambda t: t.cuda())

    def test_broadcast_batched_matmul(self):
        TestTorch._test_broadcast_batched_matmul(self, lambda t: t.cuda())

    def test_index(self):
        TestTorch._test_index(self, lambda t: t.cuda())

    def test_advancedindex(self):
        TestTorch._test_advancedindex(self, lambda t: t.cuda())

    def test_advancedindex_mixed_cpu_cuda(self):
        def test(x, ia, ib):
            self.assertEqual(x[:, ia, None, ib, 0].cpu(),
                             x.cpu()[:, ia.cpu(), None, ib.cpu(), 0])
            self.assertEqual(x[ia], x.cpu()[ia.cpu()])

        # Index cpu tensor with cuda tensor
        x = torch.randn(3, 4, 4, 4, 3)
        ia = torch.cuda.LongTensor([0, 2, 1])
        ib = torch.cuda.LongTensor([0, 2, 1])
        test(x, ia, ib)

        # Index cuda tensor with cpu tensor
        x = x.cuda()
        ia = ia.cpu()
        ib = ib.cpu()
        test(x, ia, ib)

        # Index cpu tensor with mixed cpu, cuda tensors
        x = x.cpu()
        ia = ia.cpu()
        ib = ib.cuda()
        test(x, ia, ib)

        # Index cuda tensor with mixed cpu, cuda tensors
        x = x.cuda()
        ia = ia.cpu()
        ib = ib.cuda()
        test(x, ia, ib)

    def test_advancedindex_big(self):
        TestTorch._test_advancedindex_big(self, lambda t: t.cuda())

    def test_btrifact(self):
        TestTorch._test_btrifact(self, lambda t: t.cuda())

    def test_btrisolve(self):
        TestTorch._test_btrisolve(self, lambda t: t.cuda())

    def test_dim_reduction(self):
        TestTorch._test_dim_reduction(self, lambda t: t.cuda())

    def test_tensor_gather(self):
        TestTorch._test_gather(self, lambda t: t.cuda(), False)

    def test_tensor_scatter(self):
        TestTorch._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', test_bounds=False)

    def test_tensor_scatterAdd(self):
        TestTorch._test_scatter_base(self, lambda t: t.cuda(), 'scatter_add_', test_bounds=False)

    def test_tensor_scatterFill(self):
        TestTorch._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', True, test_bounds=False)

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

    def test_int_pow(self):
        TestTorch._test_int_pow(self, lambda x: x.cuda())

    def test_remainder_overflow(self):
        TestTorch._test_remainder_overflow(self, dtype=torch.int64, device='cuda')

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

    @unittest.skipIf(not HAS_MAGMA, "no MAGMA library detected")
    def test_symeig(self):
        # Small case
        tensor = torch.randn(3, 3).cuda()
        tensor = torch.mm(tensor, tensor.t())
        eigval, eigvec = torch.symeig(tensor, eigenvectors=True)
        self.assertEqual(tensor, torch.mm(torch.mm(eigvec, eigval.diag()), eigvec.t()))

        # Large case
        tensor = torch.randn(257, 257).cuda()
        tensor = torch.mm(tensor, tensor.t())
        eigval, eigvec = torch.symeig(tensor, eigenvectors=True)
        self.assertEqual(tensor, torch.mm(torch.mm(eigvec, eigval.diag()), eigvec.t()))

    def test_arange(self):
        for t in ['IntTensor', 'LongTensor', 'FloatTensor', 'DoubleTensor']:
            a = torch.cuda.__dict__[t]()
            torch.arange(0, 10, out=a)
            b = torch.__dict__[t]()
            torch.arange(0, 10, out=b)
            self.assertEqual(a, b.cuda())

    def test_diagonal(self):
        TestTorch._test_diagonal(self, dtype=torch.float32, device='cuda')

    def test_diagflat(self):
        TestTorch._test_diagflat(self, dtype=torch.float32, device='cuda')

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_get_set_rng_state_all(self):
        states = torch.cuda.get_rng_state_all()
        before0 = torch.cuda.FloatTensor(100, device=0).normal_()
        before1 = torch.cuda.FloatTensor(100, device=1).normal_()
        torch.cuda.set_rng_state_all(states)
        after0 = torch.cuda.FloatTensor(100, device=0).normal_()
        after1 = torch.cuda.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, 0)
        self.assertEqual(before1, after1, 0)

    def test_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()

    def test_random_neg_values(self):
        TestTorch._test_random_neg_values(self, use_cuda=True)


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
            if len(decl) == 3:
                name, constr, arg_constr = decl
            elif len(decl) == 4:
                name, constr, arg_constr, desc = decl
            elif len(decl) == 5:
                name, constr, arg_constr, desc, type_subset = decl
            elif len(decl) == 6:
                name, constr, arg_constr, desc, type_subset, no_inplace = decl

            if t not in type_subset:
                continue

            precision = custom_precision.get(name, TestCuda.precision)
            if t == torch.HalfTensor:
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
                setattr(TestCuda,
                        test_name,
                        compare_cpu_gpu(constr, arg_constr, name_inner, t, precision))


if __name__ == '__main__':
    if HAS_CUDA:
        load_ignore_file()
        generate_tests()

    # skip TestTorch tests
    # hide in __name__ == '__main__' because we don't want this to be run when
    # someone imports test_cuda
    def load_tests(loader, tests, pattern):
        test_suite = unittest.TestSuite()
        for test_group in tests:
            for test in test_group:
                if test.__class__.__name__ == 'TestTorch':
                    continue
                test_suite.addTest(test)
        return test_suite

    run_tests()
