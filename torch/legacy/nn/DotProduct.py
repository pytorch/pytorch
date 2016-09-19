import torch
from .Module import Module
from .utils import clear

class DotProduct(Module):

    def __init__(self):
        super(DotProduct, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]
        self.buffer = None

    def updateOutput(self, input):
        input1, input2 = input[0], input[1]

        if not self.buffer:
           self.buffer = input1.new()

        torch.mul(self.buffer, input1, input2)
        torch.sum(self.output, self.buffer, 1)
        self.output.resize_(input1.size(0))
        return self.output

    def updateGradInput(self, input, gradOutput):
        v1 = input[0]
        v2 = input[1]
        not_batch = False

        if len(self.gradInput) != 2:
          self.gradInput[0] = self.gradInput[0] or input[0].new()
          self.gradInput[1] = self.gradInput[1] or input[1].new()
          self.gradInput = self.gradInput[:2]

        gw1 = self.gradInput[0]
        gw2 = self.gradInput[1]
        gw1.resize_as_(v1).copy_(v2)
        gw2.resize_as_(v2).copy_(v1)

        go = gradOutput.view(-1, 1).expand_as(v1)
        gw1.mul_(go)
        gw2.mul_(go)

        return self.gradInput

    def clearState(self):
        clear(self, 'buffer')
        return super(DotProduct, self).clearState()

