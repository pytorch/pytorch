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
    return torch._reshape_from_tensor(x, shape)
