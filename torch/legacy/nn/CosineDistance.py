import torch
from .Module import Module
from .utils import clear


class CosineDistance(Module):

    def __init__(self, ):
        super(CosineDistance, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]

        self._input1 = None
        self._input2 = None
        self.buffer = None
        self.w1 = None
        self.w22 = None
        self.w = None
        self.w32 = None
        self.ones = None

    def _makeContiguous(self, input1, input2):
        if not input1.is_contiguous():
            if self._input1 is None:
                self._input1 = input1.new()
            self._input1.resize_as_(input1).copy_(input1)
            input1 = self._input1

        if not input2.is_contiguous():
            if self._input2 is None:
                self._input2 = input2.new()
            self._input2.resize_as_(input2).copy_(input2)
            input2 = self._input2

        return input1, input2

    def updateOutput(self, input):
        input1, input2 = input[0], input[1]
        input1, input2 = self._makeContiguous(input1, input2)

        if self.buffer is None:
            self.buffer = input1.new()
            self.w1 = input1.new()
            self.w22 = input1.new()
            self.w = input1.new()
            self.w32 = input1.new()
            self.ones = input1.new()

        torch.mul(input1, input2, out=self.buffer)
        torch.sum(self.buffer, 1, out=self.w1)

        epsilon = 1e-12
        torch.mul(input1, input1, out=self.buffer)
        torch.sum(self.buffer, 1, out=self.w22).add_(epsilon)
        self.w22.reciprocal_()
        self.w.resize_as_(self.w22).copy_(self.w22)

        torch.mul(input2, input2, out=self.buffer)
        torch.sum(self.buffer, 1, out=self.w32).add_(epsilon)
        self.w32.reciprocal_()
        self.w.mul_(self.w32)
        self.w.sqrt_()

        torch.mul(self.w1, self.w, out=self.output)
        self.output.resize_(input1.size(0))

        return self.output

    def updateGradInput(self, input, gradOutput):
        v1 = input[0]
        v2 = input[1]
        v1, v2 = self._makeContiguous(v1, v2)

        if len(self.gradInput) != 2:
            if self.gradInput[0] is None:
                self.gradInput[0] = v1.new()
            if self.gradInput[1] is None:
                self.gradInput[1] = v1.new()
            self.gradInput = self.gradInput[:2]

        gw1 = self.gradInput[0]
        gw2 = self.gradInput[1]
        gw1.resize_as_(v1).copy_(v2)
        gw2.resize_as_(v1).copy_(v1)

        torch.mul(self.w1, self.w22, out=self.buffer)
        gw1.addcmul_(-1, self.buffer.expand_as(v1), v1)
        gw1.mul_(self.w.expand_as(v1))

        torch.mul(self.w1, self.w32, out=self.buffer)
        gw2.addcmul_(-1, self.buffer.expand_as(v1), v2)
        gw2.mul_(self.w.expand_as(v1))

        go = gradOutput.contiguous().view(-1, 1).expand_as(v1)
        gw1.mul_(go)
        gw2.mul_(go)

        return self.gradInput

    def clearState(self):
        clear(self, [
            'buffer',
            'w1',
            'w22',
            'w',
            'w32',
            'ones',
        ])
        return super(CosineDistance, self).clearState()
