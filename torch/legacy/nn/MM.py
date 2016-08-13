import torch
from .Module import Module

class MM(Module):

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
            self.output.resize_(a.size(0), b.size(1))
            torch.mm(self.output, a, b)
        else:
            if self.transA:
                a = a.transpose(2, 3)
            if self.transB:
                b = b.transpose(2, 3)

            self.output.resize_(a.size(0), a.size(1), b.size(2))
            torch.bmm(self.output, a, b)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput[0] = self.gradInput[0] or input[0].new()
        self.gradInput[1] = self.gradInput[1] or input[1].new()

        assert len(input) == 2
        a, b = input
        self.gradInput[0].resizeAs_(a)
        self.gradInput[1].resizeAs_(b)

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
            getattr(torch, f)(self.gradInput[0], b, gradOutput.transpose(h_dim, w_dim))
        else:
            getattr(torch, f)(self.gradInput[0], gradOutput, b)

        if self.transB:
            getattr(torch, f)(self.gradInput[1], gradOutput.transpose(h_dim, w_dim), a)
        else:
            getattr(torch, f)(self.gradInput[1], a, gradOutput)

        return self.gradInput

