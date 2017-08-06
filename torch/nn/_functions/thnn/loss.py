import torch
from torch._thnn import type2backend
from torch.autograd import Function

from . import _all_functions
from .auto import _BCELoss
import warnings


def _resize_weight(ctx, target):
    ctx.old_weight = ctx.weight
    if ctx.weight is not None and target.dim() != 1:
        ctx.weight = ctx.weight.view(1, target.size(1)).expand_as(target)


def _unresize_weight(ctx):
    ctx.weight = ctx.old_weight
    del ctx.old_weight


# TODO: move this code to THNN and remove _BCELoss from auto.py
class BCELoss(_BCELoss):

    @staticmethod
    def forward(ctx, input, target, weight, *args):
        if not target.is_same_size(input):
            warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                          "Please ensure they have the same size.".format(target.size(), input.size()))
        assert input.nelement() == target.nelement()
        ctx.weight = weight
        _resize_weight(ctx, target)
        result = _BCELoss.forward(ctx, input, target, ctx.weight, *args)
        _unresize_weight(ctx)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        target = ctx.saved_tensors[1]
        _resize_weight(ctx, target)
        result = _BCELoss.backward(ctx, grad_output)
        _unresize_weight(ctx)
        return result


_all_functions.append(BCELoss)
