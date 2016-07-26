import torch
from torch.legacy import nn

class CAddTable(nn.Module):
    def __init__(self, ip=False):
        super(CAddTable, self).__init__()
        self.inplace = ip
        self.gradInput = []


    def updateOutput(self, input):
        if self.inplace:
           self.output.set(input[0])
        else:
           self.output.resizeAs(input[0]).copy(input[0])

        for i in range(1, len(input)):
           self.output.add(input[i])

        return self.output


    def updateGradInput(self, input, gradOutput):
        for i in range(len(input)):
           self.gradInput[i] = self.gradInput[i] or input[0].new()
           if self.inplace:
              self.gradInput[i].set(gradOutput)
           else:
              self.gradInput[i].resizeAs(input[i]).copy(gradOutput)

        for i in range(len(input), len(self.gradInput)):
            self.gradInput[i] = nil

        return self.gradInput

