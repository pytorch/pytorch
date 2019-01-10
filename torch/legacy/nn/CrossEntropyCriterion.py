import torch
from .Criterion import Criterion
from .LogSoftMax import LogSoftMax
from .ClassNLLCriterion import ClassNLLCriterion


class CrossEntropyCriterion(Criterion):

    def __init__(self, weights=None):
        super(CrossEntropyCriterion, self).__init__()
        self.lsm = LogSoftMax()
        self.nll = ClassNLLCriterion(weights)

    def updateOutput(self, input, target):
        input = input.squeeze()
        target = target.squeeze()
        self.lsm.updateOutput(input)
        self.nll.updateOutput(self.lsm.output, target)
        self.output = self.nll.output
        return self.output

    def updateGradInput(self, input, target):
        size = input.size()
        input = input.squeeze()
        target = target.squeeze()
        self.nll.updateGradInput(self.lsm.output, target)
        self.lsm.updateGradInput(input, self.nll.gradInput)
        self.gradInput = self.lsm.gradInput.view(size)
        return self.gradInput
