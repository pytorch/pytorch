import torch
import numpy as np
import unittest

from common_utils import TestCase


HANDLED_FUNCTIONS = {}
HANDLED_FUNCTIONS_SUB = {}
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function.__name__] = func
        return func
    return decorator

def implements_sub(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[torch_function.__name__] = func
        return func
    return decorator

def implements_sub_diagonal(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function.__name__] = func
        return func
    return decorator

class DiagonalTensor:
    """A class with __torch_function__ and a specific diagonal representation"""
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs is None):
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False


@implements(torch.unique)
def unique_diag(mat1):
    "Implementation of torch.unique for DiagonalTensor objects"
    return torch.Tensor([0, mat1._i])

@implements(torch.mean)
def mean(mat1):
    "Implementation of torch.mean for DiagonalTensor objects"
    return mat1._i / mat1._N

@implements(torch.mm)
def mm(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 0

class SubTensor(torch.Tensor):
    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

@implements_sub(torch.mm)
def mm_sub(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 0

class SubDiagonalTensor(DiagonalTensor):
    """A class with __torch_function__ and a specific diagonal representation
    SubDiagonalTensor is a subclass of DiagonalTensor. All results should be scaled
    by a factor of 10."""
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "SubDiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB_DIAGONAL:
            return NotImplemented
        # In this case _torch_function_ should override DiagonalTensor objects
        return HANDLED_FUNCTIONS_SUB_DIAGONAL[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False

@implements_sub_diagonal(torch.mean)
def mean_sub(mat1):
    "Implementation of torch.mean for SubDiagonalTensor objects"
    return 10 * mat1._i / mat1._N

@implements_sub_diagonal(torch.mm)
def mm_sub_diag(mat1, mat2):
    "Implementation of torch.mm for SubDiagonalTensor objects"
    return 1

class TestOverride(TestCase):

    def test_mean(self):
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        self.assertEqual(t1.tensor(), t2)
        self.assertEqual(torch.mean(t1), torch.mean(t2))

class TestOverrideSubTensor(TestCase):

    def test_mm(self):
        t = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.mm(t, t), 0)


class TestOverrideSubDiagonalTensor(TestCase):

    def test_mean(self):
        t1 = SubDiagonalTensor(5, 2)
        t2 = 10 * torch.eye(5) * 2
        self.assertEqual(t1.tensor() * 10, t2)
        self.assertEqual(torch.mean(t1), torch.mean(t2))

    def test_mm(self):
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        t3 = torch.eye(5) * 2
        self.assertEqual(torch.mm(t1, t2), 1)
        self.assertEqual(torch.mm(t2, t1), 1)
        self.assertEqual(torch.mm(t3, t1), 0)
        self.assertEqual(torch.mm(t1, t3), 0)
        self.assertEqual(torch.mm(t3, t2), 1)
        self.assertEqual(torch.mm(t2, t3), 1)





HANDLED_FUNCTIONS_TENSOR_LIKE = {}

def implements_tensor_like(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function.__name__] = func
        return func
    return decorator

class TensorLike:
    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override DiagonalTensor objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

@implements_tensor_like(torch.dot)
def dot_override(mat1, mat2):
    return 0

@implements_tensor_like(torch.addr)
def addr_override(input, vec1, vec2, beta=1, alpha=1, out=None):
    return 0

@implements_tensor_like(torch.addmv)
def addmv_override(input, mat, vec, beta=1, alpha=1, out=None):
    return 0

@implements_tensor_like(torch.addmm)
def addmm_override(input, mat1, mat2, beta=1, alpha=1, out=None):
    return 0

@implements_tensor_like(torch.sin)
def sin_override(input, out=None):
    return -1

@implements_tensor_like(torch.sinh)
def sinh_override(input, out=None):
    return -1

@implements_tensor_like(torch.lgamma)
def lgamma_override(input, out=None):
    return -1

@implements_tensor_like(torch.mvlgamma)
def mvlgamma_override(input, p):
    return -1

@implements_tensor_like(torch.asin)
def asin_override(input, out=None):
    return -1

@implements_tensor_like(torch.cos)
def cos_override(input, out=None):
    return -1

@implements_tensor_like(torch.cosh)
def cosh_override(input, out=None):
    return -1

@implements_tensor_like(torch.acos)
def acos_override(input, out=None):
    return -1

@implements_tensor_like(torch.tan)
def tan_override(input, out=None):
    return -1

@implements_tensor_like(torch.tanh)
def tanh_override(input, out=None):
    return -1

@implements_tensor_like(torch.atan)
def atan_override(input, out=None):
    return -1

@implements_tensor_like(torch.log)
def log_override(input, out=None):
    return -1

@implements_tensor_like(torch.log10)
def log10_override(input, out=None):
    return -1

@implements_tensor_like(torch.log1p)
def log1p_override(input, out=None):
    return -1

@implements_tensor_like(torch.log2)
def log2_override(input, out=None):
    return -1

@implements_tensor_like(torch.logical_not)
def logical_not_override(input, out=None):
    return -1

@implements_tensor_like(torch.logical_xor)
def logical_xor_override(input, other, out=None):
    return -1


@implements_tensor_like(torch.sqrt)
def sqrt_override(input, out=None):
    return -1

@implements_tensor_like(torch.erf)
def erf_override(input, out=None):
    return -1

@implements_tensor_like(torch.erfc)
def erfc_override(input, out=None):
    return -1

@implements_tensor_like(torch.exp)
def exp_override(input, out=None):
    return -1

@implements_tensor_like(torch.expm1)
def expm1_override(input, out=None):
    return -1

@implements_tensor_like(torch.floor)
def floor_override(input, out=None):
    return -1

@implements_tensor_like(torch.ceil)
def ceil_override(input, out=None):
    return -1

@implements_tensor_like(torch.rsqrt)
def rsqrt_override(input, out=None):
    return -1

@implements_tensor_like(torch.sigmoid)
def sigmoid_override(input, out=None):
    return -1

@implements_tensor_like(torch.sign)
def sign_override(input, out=None):
    return -1

@implements_tensor_like(torch.frac)
def frac_override(input, out=None):
    return -1

@implements_tensor_like(torch.lerp)
def lerp_override(input, end, weight, out=None):
    return -1

@implements_tensor_like(torch.trunc)
def trunc_override(input, out=None):
    return -1

@implements_tensor_like(torch.round)
def round_override(input, out=None):
    return -1

@implements_tensor_like(torch.max)
def max_override(input, out=None):
    return -1

@implements_tensor_like(torch.min)
def min_override(input, out=None):
    return -1

@implements_tensor_like(torch.logsumexp)
def logsumexp_override(input, dim, keepdim=False, out=None):
    return -1

@implements_tensor_like(torch.where)
def where_override(condition, x, y):
    return -1

@implements_tensor_like(torch.sub)
def sub_override(input, other, out=None):
    return -1

@implements_tensor_like(torch.reciprocal)
def reciprocal_override(input, out=None):
    return -1

@implements_tensor_like(torch.remainder)
def remainder_override(input, other, out=None):
    return -1

@implements_tensor_like(torch.div)
def div_override(input, other, out=None):
    return -1

@implements_tensor_like(torch.fmod)
def fmod_override(input, other, out=None):
    return -1

@implements_tensor_like(torch.einsum)
def einsum_override(equation, *operands):
    return -1

@implements_tensor_like(torch.bmm)
def bmm_override(input, mat2, out=None):
    return -1

@implements_tensor_like(torch.addbmm)
def addbmm_override(input, batch1, batch2, alpha=1, beta=1, out=None):
    return -1

@implements_tensor_like(torch.baddbmm)
def baddbmm_override(input, batch1, batch2, alpha=1, beta=1, out=None):
    return -1

@implements_tensor_like(torch.pow)
def pow_override(input, exponent, out=None):
    return -1

@implements_tensor_like(torch.neg)
def neg_override(input, out=None):
    return -1

@implements_tensor_like(torch.argmax)
def argmax_override(input):
    return -1

@implements_tensor_like(torch.argmin)
def argmin_override(input):
    return -1

@implements_tensor_like(torch.cumprod)
def cumprod_override(input, dim, out=None, dtype=None):
    return -1

@implements_tensor_like(torch.cumsum)
def cumsum_override(input, dim, out=None, dtype=None):
    return -1

@implements_tensor_like(torch.dist)
def dist_override(input, other, p=2):
    return -1

@implements_tensor_like(torch.mean)
def mean_override(input):
    return -1

@implements_tensor_like(torch.median)
def median_override(input):
    return -1

@implements_tensor_like(torch.mode)
def mode_override(input):
    return -1

@implements_tensor_like(torch.norm)
def norm_override(input, other, p=2):
    return -1

@implements_tensor_like(torch.prod)
def prod_override(input):
    return -1

@implements_tensor_like(torch.std)
def std_override(input):
    return -1

@implements_tensor_like(torch.std_mean)
def std_mean_override(input):
    return -1

@implements_tensor_like(torch.sum)
def sum_override(input):
    return -1

@implements_tensor_like(torch.unique)
def unique_override(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    return -1

@implements_tensor_like(torch.unique_consecutive)
def unique_consecutive_override(input, return_inverse=False, return_counts=False, dim=None):
    return -1

@implements_tensor_like(torch.var)
def var_override(input):
    return -1

@implements_tensor_like(torch.var_mean)
def var_override(input):
    return -1

@implements_tensor_like(torch.argsort)
def argsort_override(input):
    return -1

@implements_tensor_like(torch.sort)
def sort_override(input, dim=-1, descending=False, out=None):
    return -1

@implements_tensor_like(torch.topk)
def topk_override(input, k, dim=None, largest=True, sorted=True, out=None):
    return -1

@implements_tensor_like(torch.chunk)
def chunk_override(input, chunks, dim=0):
    return -1

@implements_tensor_like(torch.gather)
def gather_override(input, dim, index, out=None, sparse_grad=False):
    return -1

@implements_tensor_like(torch.index_select)
def index_select_override(input, dim, index, out=None):
    return -1

class TestApis(TestCase):
    def setUp(self):
        self.t1 = TensorLike()
        self.t2 = TensorLike()
        self.t3 = TensorLike()

    def test_dot(self):
        self.assertEqual(torch.dot(self.t1, self.t2), 0)

    def test_addr(self):
        self.assertEqual(torch.addr(1, self.t1, 0, self.t2, self.t3), 0)

    def test_addmv(self):
        self.assertEqual(torch.addmv(1, self.t1, 0, self.t2, self.t3), 0)

    def test_addmm(self):
        self.assertEqual(torch.addmm(1, self.t1, 0, self.t2, self.t3), 0)

    def test_allclose(self):
        pass

    def test_linear_algebra_scalar_raises(self):
        pass

    def test_sin(self):
        self.assertEqual(torch.sin(self.t1), -1)

    def test_sinh(self):
        self.assertEqual(torch.sinh(self.t1), -1)

    def test_lgamma(self):
        self.assertEqual(torch.lgamma(self.t1), -1)

    def test_mvlgamma(self):
        self.assertEqual(torch.mvlgamma(self.t1, 1), -1)

    def test_asin(self):
        self.assertEqual(torch.asin(self.t1), -1)

    def test_cos(self):
        self.assertEqual(torch.cos(self.t1), -1)

    def test_cosh(self):
        self.assertEqual(torch.cosh(self.t1), -1)

    def test_acos(self):
        self.assertEqual(torch.acos(self.t1), -1)

    def test_tan(self):
        self.assertEqual(torch.tan(self.t1), -1)

    def test_tanh(self):
        self.assertEqual(torch.tanh(self.t1), -1)

    def test_atan(self):
        self.assertEqual(torch.atan(self.t1), -1)

    def test_log(self):
        self.assertEqual(torch.log(self.t1), -1)

    def test_log10(self):
        self.assertEqual(torch.log10(self.t1), -1)

    def test_log1p(self):
        self.assertEqual(torch.log1p(self.t1), -1)

    def test_log2(self):
        self.assertEqual(torch.log2(self.t1), -1)

    def test_logical_not(self):
        self.assertEqual(torch.logical_not(self.t1), -1)

    def test_logical_xor(self):
        self.assertEqual(torch.logical_xor(self.t1, self.t2), -1)

    def test_sqrt(self):
        self.assertEqual(torch.sqrt(self.t1), -1)

    def test_erf(self):
        self.assertEqual(torch.erf(self.t1), -1)

    def test_erfc(self):
        self.assertEqual(torch.erfc(self.t1), -1)

    def test_exp(self):
        self.assertEqual(torch.exp(self.t1), -1)

    def test_expm1(self):
        self.assertEqual(torch.expm1(self.t1), -1)

    def test_floor(self):
        self.assertEqual(torch.floor(self.t1), -1)

    def test_ceil(self):
        self.assertEqual(torch.ceil(self.t1), -1)

    def test_rsqrt(self):
        self.assertEqual(torch.rsqrt(self.t1), -1)

    def test_sigmoid(self):
        self.assertEqual(torch.sigmoid(self.t1), -1)

    def test_sign(self):
        self.assertEqual(torch.sign(self.t1), -1)

    def test_frac(self):
        self.assertEqual(torch.frac(self.t1), -1)

    def test_lerp(self):
        self.assertEqual(torch.lerp(self.t1, self.t2, 2.0), -1)

    def test_trunc(self):
        self.assertEqual(torch.trunc(self.t1), -1)

    def test_round(self):
        self.assertEqual(torch.round(self.t1), -1)

    def test_max(self):
        self.assertEqual(torch.max(self.t1), -1)

    def test_min(self):
        self.assertEqual(torch.min(self.t1), -1)

    def test_logsumexp(self):
        self.assertEqual(torch.logsumexp(self.t1, 0), -1)

    def test_max_elementwise(self):
        pass

    def test_min_elementwise(self):
        pass

    def test_where_bool_tensor(self):
        self.assertEqual(torch.where(self.t1, self.t1, self.t2), -1)

    def test_sub(self):
        self.assertEqual(torch.sub(self.t1, self.t2), -1)

    def test_reciprocal(self):
        self.assertEqual(torch.reciprocal(self.t1), -1)

    def test_remainder(self):
        self.assertEqual(torch.remainder(self.t1, self.t2), -1)

    def test_div(self):
        self.assertEqual(torch.div(self.t1, self.t2), -1)

    def test_fmod(self):
        self.assertEqual(torch.fmod(self.t1, self.t2), -1)

    def test_mm(self):
        pass

    def test_bmm(self):
        self.assertEqual(torch.bmm(self.t1, self.t2), -1)

    def test_addbmm(self):
        self.assertEqual(torch.addbmm(self.t1, self.t2, self.t3), -1)

    def test_baddbmm(self):
        self.assertEqual(torch.baddbmm(self.t1, self.t2, self.t3), -1)

    def test_einsum(self):
        # self.assertEqual(torch.einsum('i,j->ij', self.t1, self.t2), -1)
        pass

    def test_neg(self):
        self.assertEqual(torch.neg(self.t1), -1)

    def test_pow(self):
        self.assertEqual(torch.pow(self.t1, 2), -1)

    def test_argmax(self):
        self.assertEqual(torch.argmax(self.t1), -1)

    def test_argmin(self):
        self.assertEqual(torch.argmin(self.t1), -1)

    def test_cumprod(self):
        self.assertEqual(torch.cumprod(self.t1, 1), -1)

    def test_cumsum(self):
        self.assertEqual(torch.cumsum(self.t1, 1), -1)

    def test_dist(self):
        self.assertEqual(torch.dist(self.t1, self.t2), -1)

    def test_mean(self):
        self.assertEqual(torch.mean(self.t1), -1)

    def test_median(self):
        self.assertEqual(torch.median(self.t1), -1)

    def test_mode(self):
        self.assertEqual(torch.mode(self.t1), -1)

    @unittest.skip("norm is pending")
    def test_norm(self):
        self.assertEqual(torch.norm(self.t1, self.t2), -1)

    def test_prod(self):
        self.assertEqual(torch.prod(self.t1), -1)

    def test_std(self):
        self.assertEqual(torch.std(self.t1), -1)

    def test_std_mean(self):
        self.assertEqual(torch.std_mean(self.t1), -1)

    def test_sum(self):
        self.assertEqual(torch.sum(self.t1), -1)

    @unittest.skip("unique is pending")
    def test_unique(self):
        self.assertEqual(torch.unique(self.t1), -1)

    @unittest.skip("unique_consecutive is pending")
    def test_unique_consecutive(self):
        self.assertEqual(torch.unique_consecutive(self.t1), -1)

    def test_var(self):
        self.assertEqual(torch.var(self.t1), -1)

    def test_var_mean(self):
        self.assertEqual(torch.var(self.t1), -1)

    def test_argsort(self):
        self.assertEqual(torch.argsort(self.t1), -1)

    def test_sort(self):
        self.assertEqual(torch.sort(self.t1), -1)

    def test_topk(self):
        self.assertEqual(torch.topk(self.t1, 1), -1)

    def test_chunk(self):
        self.assertEqual(torch.chunk(self.t1, 2), -1)

    def test_gather(self):
        self.assertEqual(torch.gather(self.t1, 0, self.t2), -1)

    def test_index_select(self):
        self.assertEqual(torch.index_select(self.t1, 0, self.t2), -1)

if __name__ == '__main__':
    unittest.main()
