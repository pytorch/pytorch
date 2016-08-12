import torch
from torch.legacy import nn

class CSubTable(nn.Module):

    def __init__(self, ):
        super(CSubTable, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        self.output.resizeAs_(input[0]).copy_(input[0])
        self.output.add_(-1, input[1])
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput[0] = self.gradInput[0] or input[0].new()
        self.gradInput[1] = self.gradInput[1] or input[1].new()
        self.gradInput[0].resizeAs_(input[0]).copy_(gradOutput)
        self.gradInput[1].resizeAs_(input[1]).copy_(gradOutput).mul_(-1)

        self.gradInput = self.gradInput[:2]
        return self.gradInput

