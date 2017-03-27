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
        assert a.ndimension() == 2 or a.ndimension() == 3
        assert a.dim() == b.dim()

        if a.ndimension() == 2:
            if self.transA:
                a = a.t()
            if self.transB:
                b = b.t()
            self.output.resize_(a.size(0), b.size(1))
            torch.mm(a, b, out=self.output)
        else:
            if self.transA:
                a = a.transpose(2, 3)
            if self.transB:
                b = b.transpose(2, 3)

            self.output.resize_(a.size(0), a.size(1), b.size(2))
            torch.bmm(a, b, out=self.output)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput[0] is None:
            self.gradInput[0] = input[0].new()
        if self.gradInput[1] is None:
            self.gradInput[1] = input[1].new()

        assert len(input) == 2
        a, b = input
        self.gradInput[0].resize_as_(a)
        self.gradInput[1].resize_as_(b)

        assert gradOutput.ndimension() == 2 or gradOutput.ndimension() == 3
        assert a.dim() == b.dim() == gradOutput.dim()

        if gradOutput.ndimension() == 2:
            h_dim, w_dim = 0, 1
            f = "mm"
        else:
            h_dim, w_dim = 1, 2
            f = "bmm"

        if self.transA == self.transB:
            a = a.transpose(h_dim, w_dim)
            b = b.transpose(h_dim, w_dim)

        if self.transA:
            getattr(torch, f)(b, gradOutput.transpose(h_dim, w_dim), out=self.gradInput[0])
        else:
            getattr(torch, f)(gradOutput, b, out=self.gradInput[0])

        if self.transB:
            getattr(torch, f)(gradOutput.transpose(h_dim, w_dim), a, out=self.gradInput[1])
        else:
            getattr(torch, f)(a, gradOutput, out=self.gradInput[1])

        return self.gradInput
