import torch
from .Module import Module


class CriterionTable(Module):

    def __init__(self, criterion):
        super(CriterionTable, self).__init__()
        self.criterion = criterion
        self.gradInput = [criterion.gradInput]

    def updateOutput(self, input):
        self.output = self.criterion.updateOutput(*input)
        return self.output

    def updateGradInput(self, input, grad_output):
        self.criterion.updateGradInput(*input)
        return self.gradInput
