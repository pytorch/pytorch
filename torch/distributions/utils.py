from collections import namedtuple
from functools import update_wrapper
from numbers import Number
import math
import torch
import torch.nn.functional as F

# This follows semantics of numpy.finfo.
_Finfo = namedtuple('_Finfo', ['eps', 'tiny'])
_FINFO = {
    torch.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
    torch.cuda.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.cuda.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.cuda.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
}


def _finfo(tensor):
    r"""
    Return floating point info about a `Tensor`:
    - `.eps` is the smallest number that can be added to 1 without being lost.
    - `.tiny` is the smallest positive number greater than zero
      (much smaller than `.eps`).

    Args:
        tensor (Tensor): tensor of floating point data.
    Returns:
        _Finfo: a `namedtuple` with fields `.eps` and `.tiny`.
    """
    return _FINFO[tensor.storage_type()]


# promote numbers to tensors of dtype torch.get_default_dtype()
def _default_promotion(v):
    return torch.tensor(v, dtype=torch.get_default_dtype())


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    if not all(torch.is_tensor(v) or isinstance(v, Number) for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
    if not all(map(torch.is_tensor, values)):
        new_tensor = _default_promotion
        for value in values:
            if torch.is_tensor(value):
                new_tensor = value.new_tensor
                break
        values = [v if torch.is_tensor(v) else new_tensor(v) for v in values]
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
    eps = _finfo(probs).eps
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


def batch_tril(bmat, diagonal=0):
    """
    Given a batch of matrices, returns the lower triangular part of each matrix, with
    the other entries set to 0. The argument `diagonal` has the same meaning as in
    `torch.tril`.
    """
    if bmat.dim() == 2:
        return bmat.tril(diagonal=diagonal)
    else:
        return bmat * torch.tril(bmat.new(*bmat.shape[-2:]).fill_(1.0), diagonal=diagonal)


class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value
