from numbers import Number

import torch
from torch.autograd import Variable

# TODO Remove this once torch.digamma is implemented.
try:
    from torch import digamma
except ImportError:
    def digamma(x):
        """Finite difference approximation of digamma."""
        eps = x * 0.01
        return (torch.lgamma(x + eps) - torch.lgamma(x - eps)) / (2 * eps)


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
