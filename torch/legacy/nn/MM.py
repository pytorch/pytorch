import torch
from torch.legacy import nn

class MM(nn.Module):

    def __init__(self, transA=False, transB=False):
        super(MM, self).__init__()
        self.transA = transA
        self.transB = transB
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        assert len(input) == 2
        a, b = input
        assert a.nDimension() == 2 or a.nDimension() == 3
        assert a.dim() == b.dim()

        if a.nDimension() == 2:
            if self.transA:
                a = a.t()
            if self.transB:
                b = b.t()
            self.output.resize(a.size(0), b.size(1))
            self.output.mm(a, b)
        else:
            if self.transA:
                a = a.transpose(2, 3)
            if self.transB:
                b = b.transpose(2, 3)

            self.output.resize(a.size(0), a.size(1), b.size(2))
            self.output.bmm(a, b)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput[0] = self.gradInput[0] or input[0].new()
        self.gradInput[1] = self.gradInput[1] or input[1].new()

        assert len(input) == 2
        a, b = input
        self.gradInput[0].resizeAs(a)
        self.gradInput[1].resizeAs(b)

        assert gradOutput.nDimension() == 2 or gradOutput.nDimension() == 3
        assert a.dim() == b.dim() == gradOutput.dim()

        if gradOutput.nDimension() == 2:
            h_dim, w_dim = 0, 1
            f = "mm"
        else:
            h_dim, w_dim = 1, 2
            f = "bmm"

        if self.transA == self.transB:
            a = a.transpose(h_dim, w_dim)
            b = b.transpose(h_dim, w_dim)

        if self.transA:
            getattr(self.gradInput[0], f)(b, gradOutput.transpose(h_dim, w_dim))
        else:
            getattr(self.gradInput[0], f)(gradOutput, b)

        if self.transB:
            getattr(self.gradInput[1], f)(gradOutput.transpose(h_dim, w_dim), a)
        else:
            getattr(self.gradInput[1], f)(a, gradOutput)

        return self.gradInput

