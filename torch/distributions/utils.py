from functools import update_wrapper
from numbers import Number

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def expand_n(v, n):
    r"""
    Cleanly expand float or Tensor or Variable parameters.
    """
    if isinstance(v, Number):
        return torch.Tensor([v]).expand(n, 1)
    else:
        return v.expand(n, *v.size())


def _broadcast_shape(shapes):
    """
    Given a list of tensor sizes, returns the size of the resulting broadcasted
    tensor.

    Args:
        shapes (list of torch.Size): list of tensor sizes
    """
    shape = torch.Size([1])
    for s in shapes:
        shape = torch._C._infer_size(s, shape)
    return shape


def broadcast_all(*values):
    """
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.Tensor` and `torch.autograd.Variable` instances are broadcasted as
        per the `broadcasting rules
        <http://pytorch.org/docs/master/notes/broadcasting.html>`_
      - numbers.Number instances (scalars) are upcast to Tensor/Variable having
        the same size and type as the first tensor passed to `values`. If all the
        values are scalars, then they are upcasted to `torch.Tensor` having size
        `(1,)`.

    Args:
        values (list of `numbers.Number`, `torch.autograd.Variable` or
        `torch.Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number`, `torch.Tensor`
            or `torch.autograd.Variable` instance
    """
    values = list(values)
    scalar_idxs = [i for i in range(len(values)) if isinstance(values[i], Number)]
    tensor_idxs = [i for i in range(len(values)) if
                   torch.is_tensor(values[i]) or isinstance(values[i], Variable)]
    if len(scalar_idxs) + len(tensor_idxs) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number, torch.Tensor or ' +
                         'torch.autograd.Variable.')
    if tensor_idxs:
        broadcast_shape = _broadcast_shape([values[i].size() for i in tensor_idxs])
        for idx in tensor_idxs:
            values[idx] = values[idx].expand(broadcast_shape)
        template = values[tensor_idxs[0]]
        for idx in scalar_idxs:
            values[idx] = template.new(template.size()).fill_(values[idx])
    else:
        for idx in scalar_idxs:
            values[idx] = torch.Tensor([values[idx]])
    return values


def _get_clamping_buffer(tensor):
    clamp_eps = 1e-6
    if isinstance(tensor, Variable):
        tensor = tensor.data
    if isinstance(tensor, (torch.DoubleTensor, torch.cuda.DoubleTensor)):
        clamp_eps = 1e-15
    return clamp_eps


def softmax(tensor):
    """
    Wrapper around softmax to make it work with both Tensors and Variables.
    TODO: Remove once https://github.com/pytorch/pytorch/issues/2633 is resolved.
    """
    if not isinstance(tensor, Variable):
        return F.softmax(Variable(tensor), -1).data
    return F.softmax(tensor, -1)


def log_sum_exp(tensor, keepdim=True):
    """
    Numerically stable implementation for the `LogSumExp` operation. The
    summing is done along the last dimension.

    Args:
        tensor (torch.Tensor or torch.autograd.Variable)
        keepdim (Boolean): Whether to retain the last dimension on summing.
    """
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()


def logits_to_probs(logits, is_binary=False):
    """
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return F.sigmoid(logits)
    return softmax(logits)


def probs_to_logits(probs, is_binary=False):
    """
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    eps = _get_clamping_buffer(probs)
    ps_clamped = probs.clamp(min=eps, max=1 - eps)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


class lazy_property(object):
    """
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
