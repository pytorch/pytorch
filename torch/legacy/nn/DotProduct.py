import torch
from torch.legacy import nn

class DotProduct(nn.Module):

    def __init__(self):
        super(DotProduct, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]
        self.buffer = None

    def updateOutput(self, input):
        input1, input2 = input[0], input[1]

        if not self.buffer:
           self.buffer = input1.new()

        self.buffer.cmul(input1, input2)
        self.output.sum(self.buffer, 1)
        self.output.resize(input1.size(0))
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
        gw1.resizeAs(v1).copy(v2)
        gw2.resizeAs(v2).copy(v1)

        go = gradOutput.view(-1, 1).expandAs(v1)
        gw1.cmul(go)
        gw2.cmul(go)

        return self.gradInput


    def clearState(self):
        nn.utils.clear(self, 'buffer')
        return super(DotProduct, self).clearState()

