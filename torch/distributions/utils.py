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


def expand_n(v, n):
    r"""
    Cleanly expand float or Tensor parameters.
    """
    if isinstance(v, Number):
        return torch.Tensor([v]).expand(n, 1)
    else:
        return v.expand(n, *v.size())


def _broadcast_shape(shapes):
    r"""
    Given a list of tensor sizes, returns the size of the resulting broadcasted
    tensor.

    Args:
        shapes (list of torch.Size): list of tensor sizes
    """
    shape = torch.Size()
    for s in shapes:
        shape = torch._C._infer_size(s, shape)
    return shape


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.Tensor` instances are broadcasted as per the `broadcasting rules
        <http://pytorch.org/docs/master/notes/broadcasting.html>`_
      - numbers.Number instances (scalars) are upcast to Tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to Tensors having size
        `(1,)`.

    Args:
        values (list of `numbers.Number` or `torch.Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.Tensor` instance
    """
    values = list(values)
    scalar_idxs = [i for i in range(len(values)) if isinstance(values[i], Number)]
    tensor_idxs = [i for i in range(len(values)) if isinstance(values[i], torch.Tensor)]
    if len(scalar_idxs) + len(tensor_idxs) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.Tensor.')
    if tensor_idxs:
        broadcast_shape = _broadcast_shape([values[i].size() for i in tensor_idxs])
        for idx in tensor_idxs:
            values[idx] = values[idx].expand(broadcast_shape)
        template = values[tensor_idxs[0]]
        for idx in scalar_idxs:
            values[idx] = template.new(template.size()).fill_(values[idx])
    else:
        for idx in scalar_idxs:
            values[idx] = torch.tensor(float(values[idx]))
    return values


def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    return value.contiguous().view(value.shape[:-dim] + (-1,)).sum(-1)


def softmax(tensor):
    r"""
    Returns the result with softmax applied to :attr:`tensor` along the last
    dimension.
    """
    return F.softmax(tensor, -1)


def log_sum_exp(tensor, keepdim=True):
    r"""
    Numerically stable implementation for the `LogSumExp` operation. The
    summing is done along the last dimension.

    Args:
        tensor (torch.Tensor)
        keepdim (Boolean): Whether to retain the last dimension on summing.
    """
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return F.sigmoid(logits)
    return softmax(logits)


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
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value
