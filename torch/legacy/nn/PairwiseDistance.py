import torch
from torch.legacy import nn

class PairwiseDistance(nn.Module):

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

        self.diff = self.diff or input[0].new()

        torch.add(self.diff, input[0], -1, input[1]).abs_()

        self.output.resize_(input[0].size(0))
        self.output.zero_()
        self.output.add_(self.diff.pow_(self.norm).sum(1))
        self.output.pow_(1./self.norm)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input[0].dim() == 2

        if len(self.gradInput) != 2:
            self.gradInput[:] = [None, None]

        self.gradInput[0] = (self.gradInput[0] or input[0].new()).resize_(input[0].size())
        self.gradInput[1] = (self.gradInput[1] or input[1].new()).resize_(input[1].size())
        self.gradInput[0].copy_(input[0])
        self.gradInput[0].add_(-1, input[1])

        if self.norm == 1:
            self.gradInput[0].sign_()
        else:
            # Note: derivative of p-norm:
            # d/dx_k(||x||_p) = (x_k * abs(x_k)^(p-2)) / (||x||_p)^(p-1)
            if self.norm > 2:
                self.gradInput[0].mul_(self.gradInput[0].abs().pow_(self.norm-2))

            self.outExpand = self.outExpand or self.output.new()
            self.outExpand.resize_(self.output.size(0), 1)
            self.outExpand.copy_(self.output)
            self.outExpand.add_(1e-6)  # Prevent divide by zero errors
            self.outExpand.pow_(-(self.norm-1))
            self.gradInput[0].mul_(self.outExpand.expand(self.gradInput[0].size(0),
                self.gradInput[0].size(1)))

        self.grad = self.grad or gradOutput.new()
        self.ones = self.ones or gradOutput.new()

        self.grad.resizeAs_(input[0]).zero_()
        self.ones.resize_(input[0].size(1)).fill_(1)

        self.grad.addr_(gradOutput, self.ones)
        self.gradInput[0].mul_(self.grad)

        self.gradInput[1].zero_().add_(-1, self.gradInput[0])
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'diff', 'outExpand', 'grad', 'ones')
        return super(PairwiseDistance, self).clearState()

