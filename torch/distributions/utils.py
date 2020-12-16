from functools import update_wrapper
from numbers import Number
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch.overrides import has_torch_function

euler_constant = 0.57721566490153286060  # Euler Mascheroni Constant


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(isinstance(v, torch.Tensor) or has_torch_function((v,)) or isinstance(v, Number)
               for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number, '
                         'torch.Tensor or objects implementing __torch_function__.')
    if not all([isinstance(v, torch.Tensor) or has_torch_function((v,)) for v in values]):
        options: Dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [v if isinstance(v, torch.Tensor) or has_torch_function((v,)) else torch.tensor(v, **options)
                      for v in values]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            torch.ones(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type: ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


def tril_matrix_to_vec(mat, diag=0):
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f'diag ({diag}) provided is outside [{-n}, {n-1}].')
    arange = torch.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec, diag=0):
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (-(1 + 2 * diag) + ((1 + 2 * diag)**2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1))**0.5) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(f'The size of last dimension is {vec.shape[-1]} which cannot be expressed as ' +
                         'the lower triangular part of a square D x D matrix.')
    n = torch.round(n).long() if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat
