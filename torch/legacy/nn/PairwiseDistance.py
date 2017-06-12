import torch
from .Module import Module
from .utils import clear


class PairwiseDistance(Module):

    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        assert p % 1 == 0
        self.gradInput = []
        self.diff = torch.Tensor()
        self.norm = p

        self.outExpand = None
        self.grad = None
        self.ones = None

    def updateOutput(self, input):
        self.output.resize_(1)
        assert input[0].dim() == 2

        if self.diff is None:
            self.diff = input[0].new()

        torch.add(input[0], -1, input[1], out=self.diff).abs_()

        self.output.resize_(input[0].size(0))
        self.output.zero_()
        self.output.add_(self.diff.pow_(self.norm).sum(1, keepdim=False))
        self.output.pow_(1. / self.norm)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input[0].dim() == 2

        if len(self.gradInput) != 2:
            self.gradInput[:] = [None, None]

        if self.gradInput[0] is None:
            self.gradInput[0] = input[0].new()
        self.gradInput[0].resize_(input[0].size())
        if self.gradInput[1] is None:
            self.gradInput[1] = input[1].new()
        self.gradInput[1].resize_(input[1].size())
        self.gradInput[0].copy_(input[0])
        self.gradInput[0].add_(-1, input[1])

        if self.norm == 1:
            self.gradInput[0].sign_()
        else:
            # Note: derivative of p-norm:
            # d/dx_k(||x||_p) = (x_k * abs(x_k)^(p-2)) / (||x||_p)^(p-1)
            if self.norm > 2:
                self.gradInput[0].mul_(self.gradInput[0].abs().pow_(self.norm - 2))

            if self.outExpand is None:
                self.outExpand = self.output.new()
            self.outExpand.resize_(self.output.size(0), 1)
            self.outExpand.copy_(self.output.view(self.output.size(0), 1))
            self.outExpand.add_(1e-6)  # Prevent divide by zero errors
            self.outExpand.pow_(-(self.norm - 1))
            self.gradInput[0].mul_(self.outExpand.expand(self.gradInput[0].size(0),
                                                         self.gradInput[0].size(1)))

        if self.grad is None:
            self.grad = gradOutput.new()
        if self.ones is None:
            self.ones = gradOutput.new()

        self.grad.resize_as_(input[0]).zero_()
        self.ones.resize_(input[0].size(1)).fill_(1)

        self.grad.addr_(gradOutput, self.ones)
        self.gradInput[0].mul_(self.grad)

        self.gradInput[1].zero_().add_(-1, self.gradInput[0])
        return self.gradInput

    def clearState(self):
        clear(self, 'diff', 'outExpand', 'grad', 'ones')
        return super(PairwiseDistance, self).clearState()
