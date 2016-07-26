import torch
from torch.legacy import nn

class CriterionTable(nn.Module):

    def __init__(self, criterion):
        super(CriterionTable, self).__init__()
        self.criterion = criterion
        self.gradInput = [criterion.gradInput]

    def updateOutput(self, input):
        self.output = self.criterion.updateOutput(*input)
        return self.output

    def updateGradInput(self, input):
        self.criterion.updateGradInput(*input)
        return self.gradInput

