import torch
from torch.legacy import nn

class CDivTable(nn.Module):
    def __init__(self, ):
        super(CDivTable, self).__init__()
        self.gradInput = []

    def updateOutput(self, input):
        self.output.resizeAs_(input[0]).copy_(input[0])
        self.output.div_(input[1])
        return self.output

    def updateGradInput(self, input, gradOutput):
        while len(self.gradInput) < 2:
            self.gradInput.append(input[0].new())
        self.gradInput[0].resizeAs_(input[0]).copy_(gradOutput).div_(input[1])
        self.gradInput[1].resizeAs_(input[1]).zero_().addcdiv_(-1, self.gradInput[0], input[1]).mul_(input[0])

        del self.gradInput[len(input):]

        return self.gradInput

