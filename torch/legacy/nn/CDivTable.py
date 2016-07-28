import torch
from torch.legacy import nn

class CDivTable(nn.Module):
    def __init__(self, ):
        super(CDivTable, self).__init__()
        self.gradInput = []

    def updateOutput(self, input):
        self.output.resizeAs(input[0]).copy(input[0])
        self.output.cdiv(input[1])
        return self.output

    def updateGradInput(self, input, gradOutput):
        while len(self.gradInput) < 2:
            self.gradInput.append(input[0].new())
        self.gradInput[0].resizeAs(input[0]).copy(gradOutput).cdiv(input[1])
        self.gradInput[1].resizeAs(input[1]).zero().addcdiv(-1, self.gradInput[0], input[1]).cmul(input[0])

        while len(self.gradInput) > len(input):
            del self.gradInput[-1]

        return self.gradInput

