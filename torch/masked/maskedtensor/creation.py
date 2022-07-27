# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from .core import MaskedTensor


# Basic factory function
def masked_tensor(data, mask, requires_grad=False):
    from maskedtensor import is_masked_tensor

    assert not is_masked_tensor(data)
    assert not is_masked_tensor(mask)
    data = data.clone().detach()
    mask = mask.clone().detach()
    return MaskedTensor(data, mask, requires_grad)


# New function as_masked_tensor with autograd support to
# convert torch.Tensor into a MaskedTensor with some user-defined
# mask.
class AsMaskedTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, mask):
        ctx.mark_non_differentiable(mask)
        ctx.save_for_backward(mask)
        return MaskedTensor(data, mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def as_masked_tensor(data, mask):
    return AsMaskedTensor.apply(data, mask)
