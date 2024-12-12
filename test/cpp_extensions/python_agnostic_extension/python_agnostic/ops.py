from typing import List

import torch
from torch import Tensor


lib = torch.library._scoped_library("python_agnostic", "FRAGMENT")
lib.define("ultra_norm(Tensor[] inputs) -> Tensor")


def ultra_norm(inputs: List[Tensor]) -> Tensor:
    """
    Computes the ultra-L2-norm of a list of tensors via computing the norm of norms.

    Assumes:
    - inputs should not be empty
    - all tensors in inputs should be on the same device and have the same dtype

    Args:
        inputs: list of torch.tensors

    Returns:
        Scalar torch.tensor of shape ()

    """
    return torch.ops.python_agnostic.ultra_norm.default(inputs)
