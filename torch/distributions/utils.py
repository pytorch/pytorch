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


def broadcast_shape(*shapes):
    r"""
    If the tensor sizes given by `*shapes` are
    `broadcastable <http://pytorch.org/docs/master/notes/broadcasting.html>`_ ,
    this returns the size of the resulting tensor. Raises `ValueError`, otherwise.

    :param tuple shapes: shapes of tensors.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and size != 1:
                raise ValueError('shape mismatch: objects cannot be broadcast to a single shape: {}'.format(
                    ' vs '.join(map(str, shapes))))
    return tuple(reversed(reversed_shape))
