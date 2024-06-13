# mypy: allow-untyped-defs
r"""This file provides a location for operators that help exporting models via onnx.

E.g. `shape_as_tensor` and `reshape_from_tensor_shape`
are to make all dynamic sizes operations traceable.

NOTE: at one point these functions were implemented differently.
Since then we have implemented these directly in ATen, so this
file is kept purely for backward-compatibility.
"""

import torch
import torch.onnx


def shape_as_tensor(x):
    """Get the shape of a tensor as a tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: A tensor of shape [len(x.shape)] containing the size of each dimension of x.

    Example:
        >>> x = torch.randn(2, 3)
        >>> shape_as_tensor(x)
        tensor([2, 3])

    """
    return torch._shape_as_tensor(x)


def reshape_from_tensor_shape(x, shape):
    return torch._reshape_from_tensor(x, shape)
