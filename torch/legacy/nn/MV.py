import torch
from .Module import Module


class MV(Module):
    """Module to perform matrix vector multiplication on two minibatch inputs,
       producing a minibatch.
    """

    def __init__(self, trans=False):
        super(MV, self).__init__()

        self.trans = trans
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        M, v = input
        assert M.ndimension() == 2 or M.ndimension() == 3

        if M.ndimension() == 2:
            assert v.ndimension() == 1
            if self.trans:
                M = M.transpose(0, 1)
            self.output.resize_(M.size(0))
            torch.mv(M, v, out=self.output)
        else:
            assert v.ndimension() == 2
            if self.trans:
                M = M.transpose(1, 2)
            self.output.resize_(M.size(0), M.size(1), 1)
            torch.bmm(M, v.view(v.size(0), v.size(1), 1), out=self.output).resize_(M.size(0), M.size(1))

        return self.output

    def updateGradInput(self, input, gradOutput):
        M, v = input
        self.gradInput[0].resize_as_(M)
        self.gradInput[1].resize_as_(v)
        gradOutput = gradOutput.contiguous()

        assert gradOutput.ndimension() == 1 or gradOutput.ndimension() == 2

        if gradOutput.ndimension() == 2:
            assert M.ndimension() == 3
            assert v.ndimension() == 2
            bdim = M.size(0)
            odim = M.size(1)
            idim = M.size(2)

            if self.trans:
                torch.bmm(v.view(bdim, odim, 1), gradOutput.view(bdim, 1, idim), out=self.gradInput[0])
                torch.bmm(M, gradOutput.view(bdim, idim, 1), out=self.gradInput[1].view(bdim, odim, 1))
            else:
                torch.bmm(gradOutput.view(bdim, odim, 1), v.view(bdim, 1, idim), out=self.gradInput[0])
                torch.bmm(M.transpose(1, 2), gradOutput.view(bdim, odim, 1), out=self.gradInput[1].view(bdim, idim, 1))
        else:
            assert M.ndimension() == 2
            assert v.ndimension() == 1

            if self.trans:
                torch.ger(v, gradOutput, out=self.gradInput[0])
                self.gradInput[1] = M * gradOutput
            else:
                torch.ger(gradOutput, v, out=self.gradInput[0])
                self.gradInput[1] = M.t() * gradOutput

        return self.gradInput
