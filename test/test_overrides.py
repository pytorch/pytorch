import torch
import numpy as np
import unittest
import six

from common_utils import TestCase


HANDLED_FUNCTIONS = {}
HANDLED_FUNCTIONS_SUB = {}
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}
HANDLED_FUNCTIONS_TENSOR_LIKE = {}

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

def implements_tensor_like(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function.__name__] = func
        return func
    return decorator

class ImplementationMeta(type):
    def __new__(mcs, names, bases, attrs, **kwargs):
        impls = attrs.get('TORCH_IMPLEMENTATIONS', ())
        decorator = attrs.get('IMPLEMENTS_DECORATOR', None)
        for impl in impls:
            decorator(impl[0])(impl[1])
        return super(ImplementationMeta, mcs).__new__(mcs, names, bases, attrs, **kwargs)

@six.add_metaclass(ImplementationMeta)
class DiagonalTensor(object):
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the `__torch_function__`
    protocol.

    DiagonalTensor represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of `DiagonalTensor` is to provide a more compact
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
    TORCH_IMPLEMENTATIONS = (
        (torch.unique, lambda mat: torch.Tensor([0, mat._i])),
        (torch.mean, lambda mat: float(mat._i) / mat._N),
        (torch.mm, lambda mat1, mat2: 0),
    )
    IMPLEMENTS_DECORATOR = implements

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

@six.add_metaclass(ImplementationMeta)
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
    functions are working correctly

    """
    TORCH_IMPLEMENTATIONS = (
        (torch.mm, lambda mat1, mat2: 0),
    )
    IMPLEMENTS_DECORATOR = implements_sub

    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

@six.add_metaclass(ImplementationMeta)
class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. This
    class provides custom implementations of ``mean`` and ``mm``,
    scaling the mean by a factor of 10 and returning 1 from ``mm``
    instead of 0 as ``DiagonalTensor`` does.
    """
    TORCH_IMPLEMENTATIONS = (
        (torch.mean, lambda mat: 10 * float(mat._i) / mat._N),
        (torch.mm, lambda mat1, mat2: 1),
    )
    IMPLEMENTS_DECORATOR = implements_sub_diagonal

    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "SubDiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=(), kwargs=None):
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

@six.add_metaclass(ImplementationMeta)
class TensorLike(object):
    """A class that emulate the full tensor API

    This class is used to explicitly test that the full torch.tensor API
    can be overriden with a class that defines __torch_function__.
    """
    TORCH_IMPLEMENTATIONS = (
        (torch.dot, lambda mat1, mat2: 0),
        (torch.addr, lambda input, vec1, vec2, beta=1, alpha=1, out=None: 0),
        (torch.addmv, lambda input, mat, vec, beta=1, alpha=1, out=None: 0),
        (torch.addmm, lambda input, mat1, mat2, beta=1, alpha=1, out=None: 0),
        (torch.sin, lambda input, out=None: -1),
        (torch.sinh, lambda input, out=None: -1),
        (torch.lgamma, lambda input, out=None: -1),
        (torch.mvlgamma, lambda input, p: -1),
        (torch.asin, lambda input, out=None: -1),
        (torch.cos, lambda input, out=None: -1),
        (torch.cosh, lambda input, out=None: -1),
        (torch.acos, lambda input, out=None: -1),
        (torch.tan, lambda input, out=None: -1),
        (torch.tanh, lambda input, out=None: -1),
        (torch.atan, lambda input, out=None: -1),
        (torch.log, lambda input, out=None: -1),
        (torch.log10, lambda input, out=None: -1),
        (torch.log1p, lambda input, out=None: -1),
        (torch.log2, lambda input, out=None: -1),
        (torch.logical_not, lambda input, out=None: -1),
        (torch.logical_xor, lambda input, other, out=None: -1),
        (torch.sqrt, lambda input, out=None: -1),
        (torch.erf, lambda input, out=None: -1),
        (torch.erfc, lambda input, out=None: -1),
        (torch.exp, lambda input, out=None: -1),
        (torch.expm1, lambda input, out=None: -1),
        (torch.floor, lambda input, out=None: -1),
        (torch.ceil, lambda input, out=None: -1),
        (torch.rsqrt, lambda input, out=None: -1),
        (torch.sigmoid, lambda input, out=None: -1),
        (torch.sign, lambda input, out=None: -1),
        (torch.frac, lambda input, out=None: -1),
        (torch.lerp, lambda input, end, weight, out=None: -1),
        (torch.trunc, lambda input, out=None: -1),
        (torch.round, lambda input, out=None: -1),
        (torch.max, lambda input, out=None: -1),
        (torch.min, lambda input, out=None: -1),
        (torch.logsumexp, lambda input, keepdim=False, out=None: -1),
        (torch.where, lambda condition, x, y: -1),
        (torch.sub, lambda input, other, out=None: -1),
        (torch.reciprocal, lambda input, out=None: -1),
        (torch.remainder, lambda input, other, out=None: -1),
        (torch.div, lambda input, other, out=None: -1),
        (torch.fmod, lambda input, other, out=None: -1),
        (torch.einsum, lambda equation, *operands: -1),
        (torch.bmm, lambda input, mat2, out=None: -1),
        (torch.addbmm, lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1),
        (torch.baddbmm, lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1),
        (torch.pow, lambda input, exponent, out=None: -1),
        (torch.neg, lambda input, out=None: -1),
        (torch.argmax, lambda input: -1),
        (torch.argmin, lambda input: -1),
        (torch.cumprod, lambda input, dim, out=None, dtype=None: -1),
        (torch.cumsum, lambda input, dim, out=None, dtype=None: -1),
        (torch.dist, lambda input, other, p=2: -1),
        (torch.mean, lambda input: -1),
        (torch.median, lambda input: -1),
        (torch.mode, lambda input: -1),
        (torch.norm, lambda input, other, p=2: -1),
        (torch.prod, lambda input: -1),
        (torch.std, lambda input: -1),
        (torch.std_mean, lambda input: -1),
        (torch.sum, lambda input: -1),
        (torch.unique, lambda input, sorted=True, return_inverse=False, return_counts=False, dim=None: -1),
        (torch.unique_consecutive, lambda input, return_inverse=False, return_counts=False, dim=None: -1),
        (torch.var, lambda input: -1),
        (torch.var_mean, lambda input: -1),
        (torch.argsort, lambda input: -1),
        (torch.sort, lambda input, dim=-1, descending=False, out=None: -1),
        (torch.topk, lambda input, dim=-1, descending=False, out=None: -1),
        (torch.chunk, lambda input, chunks, dim=0: -1),
        (torch.gather, lambda input, dim, index, out=None, sparse_grad=False: -1),
        (torch.index_select, lambda input, dim, index, out=None: -1)
    )
    IMPLEMENTS_DECORATOR = implements_tensor_like

    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override DiagonalTensor objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

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
