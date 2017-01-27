import torch
from torch._thnn import type2backend
from torch.autograd import Function

from . import _all_functions
from .auto import _BCELoss


# TODO: move this code to THNN and remove _BCELoss from auto.py
class BCELoss(_BCELoss):

    def _resize_weight(self, target):
        self.old_weight = self.weight
        if self.weight is not None and target.dim() != 1:
            self.weight = self.weight.view(1, target.size(1)).expand_as(target)

    def _unresize_weight(self):
        self.weight = self.old_weight
        del self.old_weight

    def forward(self, input, target):
        assert input.nelement() == target.nelement()
        self._resize_weight(target)
        result = super(BCELoss, self).forward(input, target)
        self._unresize_weight()
        return result

    def backward(self, grad_output):
        target = self.saved_tensors[1]
        self._resize_weight(target)
        result = super(BCELoss, self).backward(grad_output)
        self._unresize_weight()
        return result


_all_functions.append(BCELoss)
