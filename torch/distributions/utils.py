from numbers import Number

import torch


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
    scalars = [(idx, v) for idx, v in enumerate(values) if isinstance(v, Number)]
    tensors = [(idx, v) for idx, v in enumerate(values) if isinstance(v, (torch.Tensor, torch.autograd.Variable))]
    if len(scalars) + len(tensors) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number, torch.Tensor or ' +
                         'torch.autograd.Variable.')
    if tensors:
        broadcast_shape = _broadcast_shape([t.size() for _, t in tensors])
        tensors = [(idx, v.expand(broadcast_shape)) for idx, v in tensors]
        tensor_template = tensors[0][1]
    else:
        tensor_template = torch.ones(1)
    scalars = [(idx, tensor_template.new(tensor_template.size()).fill_(s)) for idx, s in scalars]
    broadcasted_tensors = scalars + tensors
    # return the input arguments in the same order
    broadcasted_tensors.sort()
    return list(zip(*broadcasted_tensors))[1]
