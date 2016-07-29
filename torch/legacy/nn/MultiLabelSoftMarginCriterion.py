import torch
from torch.legacy import nn

class MultiLabelSoftMarginCriterion(nn.Criterion):
    """
    A MultiLabel multiclass criterion based on sigmoid:

    the loss is:
    l(x, y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
    where p[i] = exp(x[i]) / (1 + exp(x[i]))

    and with weights:
    l(x, y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))


    """

    def __init__(self, weights=None):
         super(MultiLabelSoftMarginCriterion, self).__init__()
         self.lsm = nn.Sigmoid()
         self.nll = nn.BCECriterion(weights)


    def updateOutput(self, input, target):
         input = input if input.nElement() == 1 else input.squeeze()
         target = target if target.nElement() == 1 else target.squeeze()
         self.lsm.updateOutput(input)
         self.nll.updateOutput(self.lsm.output, target)
         self.output = self.nll.output
         return self.output


    def updateGradInput(self, input, target):
         size = input.size()
         input = input if input.nElement() == 1 else input.squeeze()
         target = target if target.nElement() == 1 else target.squeeze()
         self.nll.updateGradInput(self.lsm.output, target)
         self.lsm.updateGradInput(input, self.nll.gradInput)
         self.gradInput.view(self.lsm.gradInput, size)
         return self.gradInput

