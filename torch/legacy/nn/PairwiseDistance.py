import torch
from torch.legacy import nn

class PairwiseDistance(nn.Module):

    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.gradInput = []
        self.diff = torch.Tensor()
        self.norm = p

        self.diff = None
        self.outExpand = None
        self.grad = None
        self.ones = None

    def updateOutput(self, input):
        self.output.resize(1)
        assert input[0].dim() == 2

        self.diff = self.diff or input[0].new()
        self.diff.resizeAs(input[0])

        diff = self.diff.zero()
        diff.add(input[0], -1, input[1])
        diff.abs()

        self.output.resize(input[0].size(0))
        self.output.zero()
        self.output.add(diff.pow(self.norm).sum(1))
        self.output.pow(1./self.norm)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input[0].dim() == 2

        if len(self.gradInput) != 2:
            self.gradInput[:] = [None, None]

        self.gradInput[0] = (self.gradInput[0] or input[0].new()).resize(input[0].size())
        self.gradInput[1] = (self.gradInput[1] or input[1].new()).resize(input[1].size())
        self.gradInput[0].copy(input[0])
        self.gradInput[0].add(-1, input[1])

        if self.norm == 1:
            self.gradInput[0].sign()
        else:
            # Note: derivative of p-norm:
            # d/dx_k(||x||_p) = (x_k * abs(x_k)^(p-2)) / (||x||_p)^(p-1)
            if self.norm > 2:
                self.gradInput[0].cmul(self.gradInput[0].clone().abs().pow(self.norm-2))

            self.outExpand = self.outExpand or self.output.new()
            self.outExpand.resize(self.output.size(0), 1)
            self.outExpand.copy(self.output)
            self.outExpand.add(1e-6)  # Prevent divide by zero errors
            self.outExpand.pow(-(self.norm-1))
            self.gradInput[0].cmul(self.outExpand.expand(self.gradInput[0].size(0),
                self.gradInput[0].size(1)))

        self.grad = self.grad or gradOutput.new()
        self.ones = self.ones or gradOutput.new()

        self.grad.resizeAs(input[0]).zero()
        self.ones.resize(input[0].size(1)).fill(1)

        self.grad.addr(gradOutput, self.ones)
        self.gradInput[0].cmul(self.grad)

        self.gradInput[1].zero().add(-1, self.gradInput[0])
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'diff', 'outExpand', 'grad', 'ones')
        return super(PairwiseDistance, self).clearState()

