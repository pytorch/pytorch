# Copyright (c) Meta Platforms, Inc. and affiliates

from .core import MaskedTensor, is_masked_tensor

__all__ = [
    "as_masked_tensor",
    "masked_tensor",
]


""""
These two factory functions are intended to mirror
    torch.tensor - guaranteed to be a leaf node
    torch.as_tensor - differentiable constructor that preserves the autograd history
"""

def masked_tensor(data, mask, requires_grad=False):
    assert not is_masked_tensor(data)
    assert not is_masked_tensor(mask)
    return MaskedTensor(data, mask, requires_grad)

def as_masked_tensor(data, mask):
    assert not is_masked_tensor(data)
    assert not is_masked_tensor(mask)
    return MaskedTensor._from_values(data, mask)
