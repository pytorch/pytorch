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
    return torch._shape_as_tensor(x)


def reshape_from_tensor_shape(x, shape):
    """Reshape a tensor to the given shape.

    This function is used to make dynamic size operations traceable when exporting models via ONNX.
    This function is kept for backward-compatibility. It is implemented directly in ATen.

    Parameters:
        x (Tensor): the tensor to be reshaped.
        shape (Tensor): the target shape.

    Returns:
        Tensor: the reshaped tensor.
    """
    return torch._reshape_from_tensor(x, shape)
