import torch
from torch.legacy import nn

class CosineDistance(nn.Module):

    def __init__(self, ):
        super(CosineDistance, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]

        self._input1 = None
        self._input2 = None
        self.buffer = None
        self.w1  = None
        self.w22 = None
        self.w  = None
        self.w32 = None
        self.ones = None

    def _makeContiguous(self, input1, input2):
        if not input1.isContiguous():
           self._input1 = self._input1 or input1.new()
           self._input1.resizeAs(input1).copy(input1)
           input1 = self._input1

        if not input2.isContiguous():
           self._input2 = self._input2 or input2.new()
           self._input2.resizeAs(input2).copy(input2)
           input2 = self._input2

        return input1, input2


    def updateOutput(self, input):
        input1, input2 = input[0], input[1]
        input1, input2 = self._makeContiguous(input1, input2)

        if not self.buffer:
           self.buffer = input1.new()
           self.w1  = input1.new()
           self.w22 = input1.new()
           self.w  = input1.new()
           self.w32 = input1.new()
           self.ones = input1.new()

        self.buffer.cmul(input1, input2)
        self.w1.sum(self.buffer, 1)

        epsilon = 1e-12
        self.buffer.cmul(input1, input1)
        self.w22.sum(self.buffer, 1).add(epsilon)
        self.ones.resizeAs(self.w22).fill(1)
        self.w22.cdiv(self.ones, self.w22)
        self.w.resizeAs(self.w22).copy(self.w22)

        self.buffer.cmul(input2, input2)
        self.w32.sum(self.buffer, 1).add(epsilon)
        self.w32.cdiv(self.ones, self.w32)
        self.w.cmul(self.w32)
        self.w.sqrt()

        self.output.cmul(self.w1, self.w)
        self.output.resize(input1.size(0))

        return self.output


    def updateGradInput(self, input, gradOutput):
        v1  = input[0]
        v2  = input[1]
        v1, v2 = self._makeContiguous(v1, v2)

        if len(self.gradInput) != 2:
           self.gradInput[0] = self.gradInput[0] or v1.new()
           self.gradInput[1] = self.gradInput[1] or v1.new()
           self.gradInput = self.gradInput[:2]

        gw1 = self.gradInput[0]
        gw2 = self.gradInput[1]
        gw1.resizeAs(v1).copy(v2)
        gw2.resizeAs(v1).copy(v1)

        self.buffer.cmul(self.w1, self.w22)
        gw1.addcmul(-1, self.buffer.expandAs(v1), v1)
        gw1.cmul(self.w.expandAs(v1))

        self.buffer.cmul(self.w1, self.w32)
        gw2.addcmul(-1, self.buffer.expandAs(v1), v2)
        gw2.cmul(self.w.expandAs(v1))

        go = gradOutput.view(-1, 1).expandAs(v1)
        gw1.cmul(go)
        gw2.cmul(go)

        return self.gradInput


    def clearState(self):
        nn.utils.clear(self, [
           'buffer',
           'w1',
           'w22',
           'w',
           'w32',
           'ones',
        ])
        return super(CosineDistance, self).clearState()

