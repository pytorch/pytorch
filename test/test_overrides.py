import torch
import numpy as np
import unittest
import inspect
import functools

Tensor = torch.Tensor
from torch.testing._internal.common_utils import TestCase
from torch._overrides import handle_torch_function, has_torch_function

# The functions below simulate the pure-python torch functions in the
# torch.functional namespace. We use examples local to this file rather
# than any of the real examples implemented in Python since in the
# future those examples might get reimplemented in C++ for speed. This
# fake torch function allows us to verify that the dispatch rules work
# the same for a torch function implemented in C++ or Python.

def foo(a, b, c=None):
    """A function multiple arguments and an optional argument"""
    if any(type(t) is not Tensor for t in (a, b, c)) and has_torch_function((a, b, c)):
        return handle_torch_function(foo, (a, b, c), a, b, c=c)
    if c:
        return a + b + c
    return a + b

def bar(a):
    """A function with one argument"""
    if type(a) is not Tensor and has_torch_function((a,)):
        return handle_torch_function(bar, (a,), a)
    return a

def baz(a, b):
    """A function with multiple arguments"""
    if type(a) is not Tensor or type(b) is not Tensor and has_torch_function((a, b)):
        return handle_torch_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    """Used to test that errors raised in user implementations get propagated"""
    if type(a) is not Tensor and has_torch_function((a,)):
        return handle_torch_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL is a dispatch table that
# DiagonalTensor.__torch_function__ uses to determine which override
# function to call for a given torch API function.  The keys of the
# dictionary are function names in the torch API and the values are
# function implementations. Implementations are added to
# HANDLED_FUNCTION_DIAGONAL by decorating a python function with
# implements_diagonal. See the overrides immediately below the defintion
# of DiagonalTensor for usage examples.
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(torch_function):
    """Register a torch function override for DiagonalTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator. See DiagonalTensor.__torch_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_DIAGONAL[torch_function] = func
        return func
    return decorator

class DiagonalTensor(object):
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the ``__torch_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> torch.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """
    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can re-use DiagonalTensor's
    # __torch_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in self.handled_functions:
            return NotImplemented
        return self.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False

@implements_diagonal(torch.mean)
def mean(mat):
    return float(mat._i) / mat._N

@implements_diagonal(torch.mm)
def diagonal_mm(mat1, mat2):
    return 0

@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    return -1

@implements_diagonal(torch.add)
def add(mat1, mat2):
    raise ValueError

@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    return -1

@implements_diagonal(bar)
def diagonal_bar(a):
    return -1

@implements_diagonal(quux)
def diagonal_quux(a):
    raise ValueError

# The dispatch table for SubTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(torch_function):
    "Register a torch function override for SubTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[torch_function] = func
        return func
    return decorator

class SubTensor(torch.Tensor):
    """A subclass of torch.Tensor use for testing __torch_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """
    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

@implements_sub(torch.mean)
def sub_mean(mat):
    return 0

@implements_sub(torch.mm)
def sub_mm(mat1, mat2):
    return -1

@implements_sub(torch.div)
def sub_div(input, other, out=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements_sub_diagonal(torch_function):
    "Register a torch function override for SubDiagonalTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function] = func
        return func
    return decorator

class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    def __repr__(self):
        return "SubDiagonalTensor(N={}, value={})".format(self._N, self._i)


@implements_sub_diagonal(torch.mean)
def sub_diagonal_mean(mat):
    return 10 * float(mat._i) / mat._N

@implements_sub_diagonal(bar)
def sub_diagonal_bar(mat):
    return 0

@implements_sub_diagonal(torch.mm)
def sub_diagonal_mm(mat1, mat2):
    return 1

@implements_sub_diagonal(torch.div)
def sub_diagonal_div(input, other, out=None):
    return NotImplemented

@implements_sub_diagonal(foo)
def sub_diagonal_foo(a, b, c=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_TENSOR_LIKE = {}

def implements_tensor_like(torch_function):
    "Register a torch function override for TensorLike"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function] = func
        return func
    return decorator

# Functions that are publicly available in the torch API but cannot be
# overrided with __torch_function__ (usually because none of their
# arguments are tensors or tensor-likes) need an entry in this tuple.

IGNORED_TORCH_FUNCTIONS = (
    torch.typename,
    torch.is_tensor,
    torch.is_storage,
    torch.set_default_tensor_type,
    torch.set_rng_state,
    torch.get_rng_state,
    torch.manual_seed,
    torch.initial_seed,
    torch.seed,
    torch.save,
    torch.load,
    torch.set_printoptions,
    torch.fork,
    torch.get_default_dtype,
    torch.get_num_interop_threads,
    torch.get_num_threads,
    torch.import_ir_module,
    torch.import_ir_module_from_buffer,
    torch.is_anomaly_enabled,
    torch.is_grad_enabled,
    torch.merge_type_from_type_comment,
    torch.parse_ir,
    torch.parse_schema,
    torch.parse_type_comment,
    torch.set_anomaly_enabled,
    torch.set_flush_denormal,
    torch.set_num_interop_threads,
    torch.set_num_threads,
    torch.wait,
    torch.as_tensor,
    torch.from_numpy,
    torch.get_device,
    torch.tensor,
    torch.default_generator,
    torch.has_cuda,
    torch.has_cudnn,
    torch.has_lapack,
    torch.cpp,
    torch.device,
    torch.dtype,
    torch.finfo,
    torch.has_mkl,
    torch.has_mkldnn,
    torch.has_openmp,
    torch.iinfo,
    torch.memory_format,
    torch.qscheme,
    torch.set_grad_enabled,
    torch.no_grad,
    torch.enable_grad,
    torch.layout,
    torch.align_tensors,
    torch.arange,
    torch.as_strided,
    torch.bartlett_window,
    torch.blackman_window,
    torch.can_cast,
    torch.cudnn_affine_grid_generator,
    torch.cudnn_batch_norm,
    torch.cudnn_convolution,
    torch.cudnn_convolution_transpose,
    torch.cudnn_grid_sampler,
    torch.cudnn_is_acceptable,
    torch.empty,
    torch.empty_strided,
    torch.eye,
    torch.from_file,
    torch.full,
    torch.hamming_window,
    torch.hann_window,
    torch.linspace,
    torch.logspace,
    torch.mkldnn_adaptive_avg_pool2d,
    torch.mkldnn_convolution,
    torch.mkldnn_convolution_backward_weights,
    torch.mkldnn_max_pool2d,
    torch.ones,
    torch.promote_types,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.range,
    torch.sparse_coo_tensor,
    torch.zeros,
)

# Every function in the torch API that can be overriden needs an entry
# in this tuple.
#
# Each element is itself a two-element tuple. The first entry is the
# function in the torch API to override, the second is a lambda function
# that returns -1 whose non-default positional arguments match the
# signature of the torch function in the first entry.
#
# The machinery below will call this function on a TensorLike or set of
# TensorLike objects that match the API of the lambda function and
# verify that we get -1 back from the torch API, verifying that
# __torch_function__ dispatch works correctly for the torch function.
TENSOR_LIKE_TORCH_IMPLEMENTATIONS = (
    (torch.abs, lambda input, out=None: -1),
    (torch.adaptive_avg_pool1d, lambda input, output_size: -1),
    (torch.adaptive_max_pool1d, lambda inputs, output_size: -1),
    (torch.acos, lambda input, out=None: -1),
    (torch.add, lambda input, other, out=None: -1),
    (torch.addbmm, lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1),
    (torch.addcdiv, lambda input, value, tensor1, tensor2, out=None: -1),
    (torch.addcmul, lambda input, value, tensor1, tensor2, out=None: -1),
    (torch.addmm, lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1),
    (torch.addmv, lambda input, mat, vec, beta=1, alpha=1, out=None: -1),
    (torch.addr, lambda input, vec1, vec2, beta=1, alpha=1, out=None: -1),
    (torch.affine_grid_generator, lambda theta, size, align_corners: -1),
    (torch.all, lambda input: -1),
    (torch.allclose, lambda input, other, trol=1e-05, atol=1e-08, equal_nan=False: -1),
    (torch.alpha_dropout, lambda input, p, train, inplace=False: -1),
    (torch.angle, lambda input, out=None: -1),
    (torch.any, lambda input, dim, keepdim=False, out=None: -1),
    (torch.argmax, lambda input: -1),
    (torch.argmin, lambda input: -1),
    (torch.argsort, lambda input: -1),
    (torch.asin, lambda input, out=None: -1),
    (torch.atan, lambda input, out=None: -1),
    (torch.atan2, lambda input, other, out=None: -1),
    (torch.avg_pool1d, lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True: -1),
    (torch.baddbmm, lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1),
    (torch.batch_norm, lambda input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled: -1),
    (torch.batch_norm_backward_elemt, lambda grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu: -1),
    (torch.batch_norm_backward_reduce, lambda grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g: -1),
    (torch.batch_norm_elemt, lambda input, weight, bias, mean, invstd, eps: -1),
    (torch.batch_norm_gather_stats, lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1),
    (torch.batch_norm_gather_stats_with_counts, lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1),
    (torch.batch_norm_stats, lambda input, eps: -1),
    (torch.batch_norm_update_stats, lambda input, running_mean, running_var, momentum: -1),
    (torch.bernoulli, lambda input, generator=None, out=None: -1),
    (torch.bilinear, lambda input1, input2, weight, bias: -1),
    (torch.binary_cross_entropy_with_logits, lambda input, target, weight=None, size_average=None, reduce=None, reduction='mean',
     pos_weight=None: -1),
    (torch.bincount, lambda input, weights=None, minlength=0: -1),
    (torch.bitwise_not, lambda input, out=None: -1),
    (torch.bitwise_xor, lambda input, other, out=None: -1),
    (torch.bmm, lambda input, mat2, out=None: -1),
    (torch.broadcast_tensors, lambda *tensors: -1),
    (torch.cartesian_prod, lambda *tensors: -1),
    (torch.cat, lambda tensors, dim=0, out=None: -1),
    (torch.cdist, lambda x1, c2, p=2, compute_mode=None: -1),
    (torch.ceil, lambda input, out=None: -1),
    (torch.celu, lambda input, alhpa=1., inplace=False: -1),
    (torch.chain_matmul, lambda *matrices: -1),
    (torch.cholesky, lambda input, upper=False, out=None: -1),
    (torch.cholesky_inverse, lambda input, upper=False, out=None: -1),
    (torch.cholesky_solve, lambda input1, input2, upper=False, out=None: -1),
    (torch.chunk, lambda input, chunks, dim=0: -1),
    (torch.clamp, lambda input, min, max, out=None: -1),
    (torch.clamp_min, lambda input, min, out=None: -1),
    (torch.clamp_max, lambda input, max, out=None: -1),
    (torch.clone, lambda input: -1),
    (torch.combinations, lambda input, r=2, with_replacement=False: -1),
    (torch.conj, lambda input, out=None: -1),
    (torch.constant_pad_nd, lambda input, pad, value=0: -1),
    (torch.conv1d, lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1),
    (torch.conv2d, lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1),
    (torch.conv3d, lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1),
    (torch.convolution, lambda input, weight, bias, stride, padding, dilation, transposed, output_adding, groups: -1),
    (torch.conv_tbc, lambda input, weight, bias, pad=0: -1),
    (torch.conv_transpose1d, lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1),
    (torch.conv_transpose2d, lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1),
    (torch.conv_transpose3d, lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1),
    (torch.cos, lambda input, out=None: -1),
    (torch.cosine_embedding_loss, lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean': -1),
    (torch.cosh, lambda input, out=None: -1),
    (torch.cosine_similarity, lambda x1, x2, dim=1, eps=1e-8: -1),
    (torch.cross, lambda input, other, dim=-1, out=None: -1),
    (torch.ctc_loss, lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False: -1),
    (torch.cumprod, lambda input, dim, out=None, dtype=None: -1),
    (torch.cumsum, lambda input, dim, out=None, dtype=None: -1),
    (torch.dequantize, lambda input: -1),
    (torch.det, lambda input: -1),
    (torch.detach, lambda input: -1),
    (torch.diag, lambda input, diagonal=0, out=None: -1),
    (torch.diag_embed, lambda input, diagonal=0, out=None: -1),
    (torch.diagflat, lambda input, offset=0: -1),
    (torch.diagonal, lambda input, offset=0, dim1=0, dim2=1: -1),
    (torch.digamma, lambda input, out=None: -1),
    (torch.dist, lambda input, other, p=2: -1),
    (torch.div, lambda input, other, out=None: -1),
    (torch.dot, lambda mat1, mat2: -1),
    (torch.dropout, lambda input, p, train, inplace=False: -1),
    (torch.dsmm, lambda input, mat2: -1),
    (torch.hsmm, lambda mat1, mat2: -1),
    (torch.eig, lambda input, eigenvectors=False, out=None: -1),
    (torch.einsum, lambda equation, *operands: -1),
    (torch.einsum, lambda equation, *operands: -1),
    (torch.embedding, lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
     sparse=False: -1),
    (torch.embedding_bag, lambda input, weight, offsets, max_norm=None, norm_type=2, scale_grad_by_freq=False,
     mode='mean', sparse=False, per_sample_weights=None: -1),
    (torch.empty_like, lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1),
    (torch.eq, lambda input, other, out=None: -1),
    (torch.equal, lambda input, other: -1),
    (torch.erf, lambda input, out=None: -1),
    (torch.erfc, lambda input, out=None: -1),
    (torch.erfinv, lambda input, out=None: -1),
    (torch.exp, lambda input, out=None: -1),
    (torch.expm1, lambda input, out=None: -1),
    (torch.fake_quantize_per_channel_affine, lambda input, scale, zero_point, axis, quant_min, quant_max: -1),
    (torch.fake_quantize_per_tensor_affine, lambda input, scale, zero_point, quant_min, quant_max: -1),
    (torch.fbgemm_linear_fp16_weight, lambda input, packed_weight, bias: -1),
    (torch.fbgemm_linear_fp16_weight_fp32_activation, lambda input, packed_weight, bias: -1),
    (torch.fbgemm_linear_int8_weight, lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1),
    (torch.fbgemm_linear_int8_weight_fp32_activation, lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point,
     bias: -1),
    (torch.fbgemm_linear_quantize_weight, lambda input: -1),
    (torch.fbgemm_pack_gemm_matrix_fp16, lambda input: -1),
    (torch.fbgemm_pack_quantized_matrix, lambda input, K, N: -1),
    (torch.feature_alpha_dropout, lambda input, p, train: -1),
    (torch.feature_dropout, lambda input, p, train: -1),
    (torch.fft, lambda input, signal_ndim, normalized=False: -1),
    (torch.flatten, lambda input, start_dim=0, end_dim=-1: -1),
    (torch.flip, lambda input, dims: -1),
    (torch.frobenius_norm, lambda input, dim=None, keepdim=False, out=None: -1),
    (torch.floor, lambda input, out=None: -1),
    (torch.fmod, lambda input, other, out=None: -1),
    (torch.frac, lambda input, out=None: -1),
    (torch.full_like, lambda input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False: -1),
    (torch.gather, lambda input, dim, index, out=None, sparse_grad=False: -1),
    (torch.ge, lambda input, other, out=None: -1),
    (torch.geqrf, lambda input, out=None: -1),
    (torch.ger, lambda input, vec2, out=None: -1),
    (torch.grid_sampler, lambda input, grid, interpolation_mode, padding_mode, align_corners: -1),
    (torch.grid_sampler_2d, lambda input, grid, interpolation_mode, padding_mode, align_corners: -1),
    (torch.grid_sampler_3d, lambda input, grid, interpolation_mode, padding_mode, align_corners: -1),
    (torch.group_norm, lambda input, num_groups, weight=None, bias=None, eps=1e-05: -1),
    (torch.gru, lambda input, hx, params, has_biases, num_layers, gropout, train, bidirectional, batch_first: -1),
    (torch.gru_cell, lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1),
    (torch.gt, lambda input, other, out=None: -1),
    (torch.hardshrink, lambda input, lambd=0.5: -1),
    (torch.hinge_embedding_loss, lambda input, target, margin=1.0, size_average=None, reduce=None, reduction='mean': -1),
    (torch.histc, lambda input, bins=100, min=0, max=0, out=None: -1),
    (torch.hspmm, lambda mat1, mat2, out=None: -1),
    (torch.ifft, lambda input, signal_ndim, normalized=False: -1),
    (torch.imag, lambda input, out=None: -1),
    (torch.index_add, lambda input, dim, index, source: -1),
    (torch.index_copy, lambda input, dim, index, source: -1),
    (torch.index_put, lambda input, indices, values, accumulate=False: -1),
    (torch.index_select, lambda input, dim, index, out=None: -1),
    (torch.index_fill, lambda input, dim, index, value: -1),
    (torch.isfinite, lambda tensor: -1),
    (torch.isinf, lambda tensor: -1),
    (torch.instance_norm, lambda input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, cudnn_enabled: -1),
    (torch.int_repr, lambda input: -1),
    (torch.inverse, lambda input, out=None: -1),
    (torch.irfft, lambda input, signal_ndim, normalized=False, onesided=True, signal_sizes=None: -1),
    (torch.is_complex, lambda input: -1),
    (torch.is_distributed, lambda input: -1),
    (torch.is_floating_point, lambda input: -1),
    (torch.is_nonzero, lambda input: -1),
    (torch.is_same_size, lambda input, other: -1),
    (torch.is_signed, lambda input: -1),
    (torch.isclose, lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1),
    (torch.isnan, lambda input: -1),
    (torch.kl_div, lambda input, target, size_average=None, reduce=None, reduction='mean': -1),
    (torch.kthvalue, lambda input, k, dim=None, keepdim=False, out=None: -1),
    (torch.layer_norm, lambda input, normalized_shape, weight=None, bias=None, esp=1e-05: -1),
    (torch.le, lambda input, other, out=None: -1),
    (torch.lerp, lambda input, end, weight, out=None: -1),
    (torch.lgamma, lambda input, out=None: -1),
    (torch.log, lambda input, out=None: -1),
    (torch.log_softmax, lambda input, dim, dtype: -1),
    (torch.log10, lambda input, out=None: -1),
    (torch.log1p, lambda input, out=None: -1),
    (torch.log2, lambda input, out=None: -1),
    (torch.logdet, lambda input: -1),
    (torch.logical_not, lambda input, out=None: -1),
    (torch.logical_xor, lambda input, other, out=None: -1),
    (torch.logsumexp, lambda input, names, keepdim, out=None: -1),
    (torch.lstm, lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional: -1),
    (torch.lstm_cell, lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1),
    (torch.lstsq, lambda input, A, out=None: -1),
    (torch.lt, lambda input, other, out=None: -1),
    (torch.lu, lambda A, pivot=True, get_infos=False, out=None: -1),
    (torch.lu_solve, lambda input, LU_data, LU_pivots, out=None: -1),
    (torch.margin_ranking_loss, lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean': -1),
    (torch.masked_fill, lambda input, mask, value: -1),
    (torch.masked_scatter, lambda input, mask, source: -1),
    (torch.masked_select, lambda input, mask, out=None: -1),
    (torch.matmul, lambda input, other, out=None: -1),
    (torch.matrix_power, lambda input, n: -1),
    (torch.matrix_rank, lambda input, tol=None, symmetric=False: -1),
    (torch.max, lambda input, out=None: -1),
    (torch.max_pool1d, lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1),
    (torch.max_pool2d, lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1),
    (torch.max_pool3d, lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1),
    (torch.max_pool1d_with_indices, lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
     ceil_mode=False: -1),
    (torch.mean, lambda input: -1),
    (torch.median, lambda input: -1),
    (torch.meshgrid, lambda *tensors, **kwargs: -1),
    (torch.min, lambda input, out=None: -1),
    (torch.miopen_batch_norm, lambda input, weight, bias, running_mean, running_var, training, exponential_average_factor,
     epsilon: -1),
    (torch.miopen_convolution, lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1),
    (torch.miopen_convolution_transpose, lambda input, weight, bias, padding, output_padding, stride, dilation, groups, benchmark,
     deterministic: -1),
    (torch.miopen_depthwise_convolution, lambda input, weight, bias, padding, stride, dilation, groups, benchmark,
     deterministic: -1),
    (torch.miopen_rnn, lambda input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train,
     bidirectional, batch_sizes, dropout_state: -1),
    (torch.mm, lambda input, mat2, out=None: -1),
    (torch.mode, lambda input: -1),
    (torch.mul, lambda input, other, out=None: -1),
    (torch.multinomial, lambda input, num_samples, replacement=False, out=None: -1),
    (torch.mv, lambda input, vec, out=None: -1),
    (torch.mvlgamma, lambda input, p: -1),
    (torch.narrow, lambda input, dim, start, length: -1),
    (torch.native_batch_norm, lambda input, weight, bias, running_mean, running_var, training, momentum, eps: -1),
    (torch.native_layer_norm, lambda input, weight, bias, M, N, eps: -1),
    (torch.native_norm, lambda input, p=2: -1),
    (torch.ne, lambda input, other, out=None: -1),
    (torch.neg, lambda input, out=None: -1),
    (torch.nonzero, lambda input, out=None, as_tuple=False: -1),
    (torch.norm, lambda input, p='fro', dim=None, keepdim=False, out=None, dtype=None: -1),
    (torch.norm_except_dim, lambda v, pow=2, dim=0: -1),
    (torch.normal, lambda mean, std, out=None: -1),
    (torch.nuclear_norm, lambda input, p='fro', dim=None, keepdim=False, out=None, dtype=None: -1),
    (torch.numel, lambda input: -1),
    (torch.orgqr, lambda input1, input2: -1),
    (torch.ormqr, lambda input, input2, input3, left=True, transpose=False: -1),
    (torch.pairwise_distance, lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1),
    (torch.pdist, lambda input, p=2: -1),
    (torch.pinverse, lambda input, rcond=1e-15: -1),
    (torch.pixel_shuffle, lambda input, upscale_factor: -1),
    (torch.poisson, lambda input, generator=None: -1),
    (torch.poisson_nll_loss, lambda input, target, log_input, full, eps, reduction: -1),
    (torch.polygamma, lambda input, n, out=None: -1),
    (torch.prelu, lambda input, weight: -1),
    (torch.ones_like, lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1),
    (torch.pow, lambda input, exponent, out=None: -1),
    (torch.prod, lambda input: -1),
    (torch.q_per_channel_axis, lambda input: -1),
    (torch.q_per_channel_scales, lambda input: -1),
    (torch.q_per_channel_zero_points, lambda input: -1),
    (torch.q_scale, lambda input: -1),
    (torch.q_zero_point, lambda input: -1),
    (torch.qr, lambda input, some=True, out=None: -1),
    (torch.quantize_per_channel, lambda input, scales, zero_points, axis, dtype: -1),
    (torch.quantize_per_tensor, lambda input, scale, zero_point, dtype: -1),
    (torch.quantized_gru, lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional: -1),
    (torch.quantized_gru_cell, lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
     scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
    (torch.quantized_lstm, lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first,
     dtype=None, use_dynamic=False: -1),
    (torch.quantized_lstm_cell, lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
     scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
    (torch.quantized_max_pool2d, lambda input, kernel_size, stride, padding, dilation, ceil_mode=False: -1),
    (torch.quantized_rnn_relu_cell, lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
     col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
    (torch.quantized_rnn_tanh_cell, lambda input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih,
     col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh: -1),
    (torch.rand_like, lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1),
    (torch.randint_like, lambda input, low, high, dtype=None, layout=torch.strided, device=None, requires_grad=False: -1),
    (torch.randn_like, lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1),
    (torch.real, lambda input, out=None: -1),
    (torch.reciprocal, lambda input, out=None: -1),
    (torch.relu, lambda input, inplace=False: -1),
    (torch.remainder, lambda input, other, out=None: -1),
    (torch.renorm, lambda input, p, dim, maxnorm, out=None: -1),
    (torch.repeat_interleave, lambda input, repeats, dim=None: -1),
    (torch.reshape, lambda input, shape: -1),
    (torch.result_type, lambda tensor1, tensor2: -1),
    (torch.rfft, lambda input, signal_ndim, normalized=False, onesided=True: -1),
    (torch.rnn_relu, lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1),
    (torch.rnn_relu_cell, lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1),
    (torch.rnn_tanh, lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1),
    (torch.rnn_tanh_cell, lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1),
    (torch.roll, lambda input, shifts, dims=None: -1),
    (torch.rot90, lambda input, k, dims: -1),
    (torch.round, lambda input, out=None: -1),
    (torch.rrelu, lambda input, lower=1. / 8, upper=1. / 3, training=False, inplace=False: -1),
    (torch.rsqrt, lambda input, out=None: -1),
    (torch.rsub, lambda input, other, alpha=1: -1),
    (torch.saddmm, lambda input, mat1, mat2, beta, alpha, out=None: -1),
    (torch.scalar_tensor, lambda s, dtype=None, layour=None, device=None, pin_memory=None: -1),
    (torch.scatter, lambda input, dim, index, src: -1),
    (torch.scatter_add, lambda input, dim, index, src: -1),
    (torch.select, lambda input, dim, index: -1),
    (torch.selu, lambda input, inplace=False: -1),
    (torch.sigmoid, lambda input, out=None: -1),
    (torch.sign, lambda input, out=None: -1),
    (torch.sin, lambda input, out=None: -1),
    (torch.sinh, lambda input, out=None: -1),
    (torch.slogdet, lambda input: -1),
    (torch.smm, lambda input, mat2: -1),
    (torch.spmm, lambda input, mat2: -1),
    (torch.softmax, lambda input, dim, dtype=None: -1),
    (torch.solve, lambda input, A, out=None: -1),
    (torch.sort, lambda input, dim=-1, descending=False, out=None: -1),
    (torch.split, lambda tensor, split_size_or_sections, dim=0: -1),
    (torch.split_with_sizes, lambda tensor, split_size_or_sections, dim=0: -1),
    (torch.sqrt, lambda input, out=None: -1),
    (torch.squeeze, lambda input, dim=None, out=None: -1),
    (torch.sspaddmm, lambda input, mat1, mat2, beta, alpha, out=None: -1),
    (torch.stack, lambda tensors, dim=0, out=None: -1),
    (torch.std, lambda input: -1),
    (torch.std_mean, lambda input: -1),
    (torch.stft, lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect',
     normalized=False, onesided=True: -1),
    (torch.sub, lambda input, other, out=None: -1),
    (torch.sum, lambda input: -1),
    (torch.svd, lambda input, some=True, compute_uv=True, out=None: -1),
    (torch.symeig, lambda input, eigenvectors=False, upper=True, out=None: -1),
    (torch.t, lambda input: -1),
    (torch.take, lambda input, index: -1),
    (torch.tan, lambda input, out=None: -1),
    (torch.tanh, lambda input, out=None: -1),
    (torch.tensordot, lambda a, b, dims=2: -1),
    (torch.threshold, lambda input, threshold, value, inplace=False: -1),
    (torch.topk, lambda input, k, dim=-1, descending=False, out=None: -1),
    (torch.trace, lambda input: -1),
    (torch.transpose, lambda input, dim0, dim1: -1),
    (torch.trapz, lambda y, x, dim=-1: -1),
    (torch.triangular_solve, lambda input, A, upper=True, transpose=False, unitriangular=False: -1),
    (torch.tril, lambda input, diagonal=0, out=None: -1),
    (torch.tril_indices, lambda row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided: -1),
    (torch.triplet_margin_loss, lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None,
     reduce=None, reduction='mean': -1),
    (torch.triu, lambda input, diagonal=0, out=None: -1),
    (torch.triu_indices, lambda row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided: -1),
    (torch.trunc, lambda input, out=None: -1),
    (torch.unbind, lambda input, dim=0: -1),
    (torch.unique, lambda input, sorted=True, return_inverse=False, return_counts=False, dim=None: -1),
    (torch.unique_consecutive, lambda input, return_inverse=False, return_counts=False, dim=None: -1),
    (torch.unsqueeze, lambda input, dim, out=None: -1),
    (torch.var, lambda input: -1),
    (torch.var_mean, lambda input: -1),
    (torch.where, lambda condition, x, y: -1),
    (torch.zeros_like, lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1),
)

TENSOR_LIKE_OVERRIDES = tuple(t[0] for t in TENSOR_LIKE_TORCH_IMPLEMENTATIONS)

def generate_tensor_like_torch_implementations():
    torch_vars = vars(torch)
    untested_funcs = []
    for func_name in torch.__all__ + dir(torch._C._VariableFunctions):
        # ignore private functions or functions that are deleted in torch.__init__
        if func_name.startswith('_') or func_name == 'unique_dim':
            continue
        func = getattr(torch, func_name)
        # IGNORED_TORCH_FUNCTIONS are functions that are public but cannot be
        # overriden by __torch_function__
        if func in IGNORED_TORCH_FUNCTIONS:
            msg = "torch.{} is in IGNORED_TORCH_FUNCTIONS but still has an explicit override"
            assert func not in TENSOR_LIKE_OVERRIDES, msg.format(func.__name__)
            continue
        # ignore in-place operators
        if func_name.endswith('_'):
            continue
        # only consider objects with lowercase names
        if not func_name.islower():
            continue
        if func not in TENSOR_LIKE_OVERRIDES:
            untested_funcs.append("torch.{}".format(func.__name__))
    msg = (
        "The following functions are not tested for __torch_function__ "
        "support, please either add an entry in "
        "TENSOR_LIKE_TORCH_IMPLEMENTATIONS for this function or if a "
        "__torch_function__ override does not make sense, add an entry to "
        "IGNORED_TORCH_FUNCTIONS.\n\n{}"
    )

    # The assertion below is disabled temporarily while we land the rest
    # of this functionality, it will be re-enabled later when there is a
    # smaller chance that enabling this test will land simultaneously
    # with a new operator being added to the torch namespace.

    # assert len(untested_funcs) == 0, msg.format(pprint.pformat(untested_funcs))
    for func, override in TENSOR_LIKE_TORCH_IMPLEMENTATIONS:
        # decorate the overrides with implements_tensor_like
        implements_tensor_like(func)(override)

generate_tensor_like_torch_implementations()

class TensorLike(object):
    """A class that overrides the full torch API

    This class is used to explicitly test that the full torch.tensor API
    can be overriden with a class that defines __torch_function__.
    """
    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override TensorLike objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

class TestTorchFunctionOverride(TestCase):
    def test_mean(self):
        """Test that a function with one argument can be overrided"""
        t1 = DiagonalTensor(5, 2)
        t2 = SubTensor([[1, 2], [1, 2]])
        t3 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.mean(t1), 12.5)
        self.assertEqual(bar(t1), -1)
        self.assertEqual(torch.mean(t2), 0)
        self.assertEqual(bar(t2), t2)
        self.assertEqual(torch.mean(t3), 125)
        self.assertEqual(bar(t3), 0)

    def test_mm(self):
        """Test that a function with multiple arguments can be overrided"""
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        t3 = SubTensor([[1, 2], [1, 2]])
        t4 = SubDiagonalTensor(5, 2)
        # only DiagonalTensor so should always get DiagonalTensor result
        self.assertEqual(torch.mm(t1, t1), 0)
        # tensor and DiagonalTensor, always return DiagonalTensor result
        self.assertEqual(torch.mm(t1, t2), 0)
        self.assertEqual(torch.mm(t2, t1), 0)
        # only SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t3), -1)
        # tensor and SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t2), -1)
        self.assertEqual(torch.mm(t2, t3), -1)
        # DiagonalTensor and SubTensor are unrelated classes so the result
        # depends on which argument appears first
        self.assertEqual(torch.mm(t3, t1), -1)
        self.assertEqual(torch.mm(t1, t3), 0)
        # SubDiagonalTensor should take precedence over DiagonalTensor
        # but should behave otherwise the same as DiagonalTensor
        self.assertEqual(torch.mm(t4, t4), 1)
        self.assertEqual(torch.mm(t4, t1), 1)
        self.assertEqual(torch.mm(t1, t4), 1)
        self.assertEqual(torch.mm(t4, t2), 1)
        self.assertEqual(torch.mm(t2, t4), 1)
        self.assertEqual(torch.mm(t3, t4), -1)
        self.assertEqual(torch.mm(t4, t3), 0)

    def test_precedence_semantics(self):
        """Test semantics for __torch_function__ for functions that take
        multiple arugments

        For functions that take multiple arguments, the appropriate
        __torch_function__ implementation to call is determined by
        examining the types of the arguments. The precedence order is
        left-to-right in the argument list, except subclasses are always
        checked before superclasses. The first result of calling the
        implementations in precedence order that is not NotImplemented
        is returned to the user. If all implementations return
        NotImplemented, a TypeError is raised.

        All cases are tested with functions implemented in C++ and
        either foo or baz, which are python functions defined above that
        are instrumented to obey the same dispatch rules as the
        functions in torch.functional.
        """
        # DiagonalTensor has a valid override and SubDiagonal has an
        # override that returns NotImplemented so we should call the
        # DiagonalTensor implementation, returning -1
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.div(t1, t2), -1)
        self.assertEqual(torch.div(t2, t1), -1)
        self.assertEqual(foo(t1, t2), -1)
        self.assertEqual(foo(t2, t1), -1)

        # SubTensor has an implementation that returns NotImplemented as
        # well so it should behave exactly like SubDiagonalTensor in the
        # test above
        t3 = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.div(t1, t3), -1)
        self.assertEqual(torch.div(t3, t1), -1)
        self.assertEqual(foo(t1, t3), -1)
        self.assertEqual(foo(t3, t1), -1)

        # div between SubTensor and SubDiagonalTensor should raise
        # TypeError since both have an implementation that
        # explicitly returns NotImplemented
        with self.assertRaises(TypeError):
            torch.div(t2, t3)
        with self.assertRaises(TypeError):
            torch.div(t3, t2)
        with self.assertRaises(TypeError):
            foo(t2, t3)
        with self.assertRaises(TypeError):
            foo(t3, t2)

        # none of DiagonalTensor, SubdiagonalTensor, or SubTensor have a
        # mul or a baz implementation so all ops should raise TypeError
        with self.assertRaises(TypeError):
            torch.mul(t1, t1)
        with self.assertRaises(TypeError):
            torch.mul(t1, t2)
        with self.assertRaises(TypeError):
            torch.mul(t1, t3)
        with self.assertRaises(TypeError):
            torch.mul(t2, t1)
        with self.assertRaises(TypeError):
            torch.mul(t2, t2)
        with self.assertRaises(TypeError):
            torch.mul(t2, t3)
        with self.assertRaises(TypeError):
            torch.mul(t3, t1)
        with self.assertRaises(TypeError):
            torch.mul(t3, t2)
        with self.assertRaises(TypeError):
            torch.mul(t3, t3)
        with self.assertRaises(TypeError):
            baz(t1, t1)
        with self.assertRaises(TypeError):
            baz(t1, t2)
        with self.assertRaises(TypeError):
            baz(t1, t3)
        with self.assertRaises(TypeError):
            baz(t2, t1)
        with self.assertRaises(TypeError):
            baz(t2, t2)
        with self.assertRaises(TypeError):
            baz(t2, t3)
        with self.assertRaises(TypeError):
            baz(t3, t1)
        with self.assertRaises(TypeError):
            baz(t3, t2)
        with self.assertRaises(TypeError):
            baz(t3, t3)

    def test_user_implementation_raises(self):
        """Test that errors raised in user implementations propagate correctly"""
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        with self.assertRaises(ValueError):
            torch.add(t1, t2)
        with self.assertRaises(ValueError):
            quux(t1)

def generate_tensor_like_override_tests(cls):
    def test_generator(func, override):
        if torch._six.PY3:
            args = inspect.getfullargspec(override)
        else:
            args = inspect.getargspec(override)
        nargs = len(args.args)
        if args.defaults is not None:
            nargs -= len(args.defaults)
        func_args = [TensorLike() for _ in range(nargs)]
        if args.varargs is not None:
            func_args += [TensorLike(), TensorLike()]

        def test(self):
            self.assertEqual(func(*func_args), -1)

        return test

    for func, override in TENSOR_LIKE_TORCH_IMPLEMENTATIONS:
        test_method = test_generator(func, override)
        name = 'test_{}'.format(func.__name__)
        test_method.__name__ = name
        setattr(cls, name, test_method)

generate_tensor_like_override_tests(TestTorchFunctionOverride)

if __name__ == '__main__':
    unittest.main()
