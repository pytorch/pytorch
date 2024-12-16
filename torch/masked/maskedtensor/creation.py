# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Any

from .core import MaskedTensor


__all__ = [
    "as_masked_tensor",
    "masked_tensor",
]


# These two factory functions are intended to mirror
#     torch.tensor - guaranteed to be a leaf node
#     torch.as_tensor - differentiable constructor that preserves the autograd history


def masked_tensor(data: Any, mask: Any, requires_grad: bool = False) -> MaskedTensor:
    return MaskedTensor(data, mask, requires_grad)


def as_masked_tensor(data: Any, mask: Any) -> MaskedTensor:
    return MaskedTensor._from_values(data, mask)
