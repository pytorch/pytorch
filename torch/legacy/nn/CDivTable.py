import torch
from torch.legacy import nn

class CDivTable(nn.Module):
    def __init__(self, ):
        super(CDivTable, self).__init__()
        self.gradInput = []

    def updateOutput(self, input):
        self.output.resizeAs(input[1]).copy(input[1])
        self.output.cdiv(input[2])
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput[0] = self.gradInput[0] or input[0].new()
        self.gradInput[1] = self.gradInput[1] or input[0].new()
        self.gradInput[0].resizeAs(input[0]).copy(gradOutput).cdiv(input[1])
        self.gradInput[1].resizeAs(input[1]).zero().addcdiv(-1, self.gradInput[0], input[1]).cmul(input[0])

        for i in range(len(input), len(self.gradInput)):
            self.gradInput[i] = nil

        return self.gradInput

